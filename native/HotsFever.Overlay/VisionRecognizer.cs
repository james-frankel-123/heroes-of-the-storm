using System.Net.Http;
using System.Text;
using System.Text.Json;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Text;
using Windows.Graphics.DirectX;
using Colors = Microsoft.UI.Colors; // WinAppSDK's palette; yields Windows.UI.Color for Win2D

namespace HotsFever.Overlay;

public sealed class VisionDraft
{
    public string? Map { get; set; }
    public List<string> LeftTeam { get; set; } = new();   // the local player's team (always the left column)
    public List<string> RightTeam { get; set; } = new();  // the enemy team
    public List<string> BansLeft { get; set; } = new();
    public List<string> BansRight { get; set; } = new();

    // Slots the team on the clock has highlighted but NOT confirmed, plus the big
    // centre portrait. These look picked but aren't — never put them on the board.
    public List<string> PendingLeft { get; set; } = new();
    public List<string> PendingRight { get; set; } = new();
    public string? PreviewHero { get; set; }

    // Echoed by the backend so the log shows which model answered and what it cost.
    public string? Model { get; set; }
    public int PromptTokens { get; set; }
    public int CompletionTokens { get; set; }

    public bool IsEmpty => LeftTeam.Count == 0 && RightTeam.Count == 0 && BansLeft.Count == 0
        && BansRight.Count == 0 && PendingLeft.Count == 0 && PendingRight.Count == 0
        && string.IsNullOrEmpty(PreviewHero);

    // A read that found nothing at all AND no map almost certainly means we cropped the
    // wrong pixels — an early draft still shows the map banner.
    public bool FoundNothing => IsEmpty && string.IsNullOrEmpty(Map);
}

/// <summary>
/// Reads the live draft screen by sending it to the backend vision endpoint (which runs
/// it through a vision LLM).
///
/// It sends CROPS, not the whole screen. Sending the full frame put the two team columns
/// — thin strips at the far left and right edges — against a centre splash that fills
/// most of the image and renders the previewing hero's name in the largest text on
/// screen. The model anchored on that splash: it reported heroes nobody had picked, put
/// enemy heroes on our team, and once listed the same hero on both sides. Cropping to
/// fixed regions makes team identity STRUCTURAL (we know which column we cut) and keeps
/// the splash and the bottom hero pool out of the image entirely.
/// </summary>
public static class VisionRecognizer
{
    // Overridable so a branch preview deploy or a local `npm run dev` can be tested
    // against a live draft without touching production.
    private static readonly string Endpoint =
        Environment.GetEnvironmentVariable("HOTSFEVER_VISION_ENDPOINT") is { Length: > 0 } url
            ? url
            : "https://hotsfever.com/api/draft-vision";
    // Shared secret matching the Vercel DRAFT_VISION_TOKEN. Gates casual abuse of
    // our vision spend; acceptable to embed for the beta.
    private const string Token = "C6JTbssctxg7ipGxeh6jf83uRIm1Iji3";
    private const int TargetWidth = 1600; // full-frame fallback only

    // Regions of interest, as fractions of the frame HEIGHT and anchored to the screen
    // EDGES. HotS pins the draft furniture to the edges and scales it with height, so
    // height-relative + edge-anchored should carry across aspect ratios. Measured on a
    // 3440x1440 ultrawide frame and padded a little for drift.
    private const double ColWidthH = 0.34;   // column strip width
    private const double ColTopH = 0.07;
    private const double ColHeightH = 0.84;  // through the 5th slot's name banner
    // Ban row and map banner. These were first calibrated tight against one archived
    // 3440x1440 frame (bans at y 15-150, map text at y 15-50) and validated there — but a
    // live draft put both LOWER than that, so the tight crops caught nothing but starfield
    // and the model invented bans from empty space (a different set almost every read).
    // The band is now deliberately generous: a bit of extra background costs a few image
    // tokens, while missing the row entirely costs every ban in the draft.
    private const double BanInsetH = 0.08;   // ban row starts just inboard of the column
    private const double BanWidthH = 0.52;
    private const double BanHeightH = 0.26;
    private const double MapWidthH = 0.62;   // map banner, centred on the top edge
    private const double MapHeightH = 0.20;

    private const int Gutter = 40;  // black gap so the two columns can't be read as one
    private const int Header = 64;  // room for the drawn region label

    private const int EmptyReadsBeforeFallback = 3;

    /// <summary>Set by the overlay so ROI problems surface in draft-watch.log.</summary>
    public static Action<string>? Log;

    private static int _emptyReads;
    private static bool _fullFrameFallback;
    private static bool _savedFullFrame;

    /// <summary>Called when a new draft starts, so a latched fallback doesn't persist.</summary>
    public static void ResetForNewDraft()
    {
        _emptyReads = 0;
        _fullFrameFallback = false;
        _savedFullFrame = false;
    }

    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(30) };
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    public static async System.Threading.Tasks.Task<VisionDraft?> RecognizeAsync(ScreenCapture.RawFrame frame)
    {
        try
        {
            if (_fullFrameFallback) return await RecognizeFullFrameAsync(frame);

            var device = CanvasDevice.GetSharedDevice();
            using var src = CanvasBitmap.CreateFromBytes(
                device, frame.Bgra, frame.Width, frame.Height, DirectXPixelFormat.B8G8R8A8UIntNormalized);

            var teams = await RenderTeamsAsync(device, src, frame.Width, frame.Height);
            var meta = await RenderMetaAsync(device, src, frame.Width, frame.Height);
            if (teams == null || meta == null) return null;
            SaveFrame("last-vision-teams.jpg", teams);
            SaveFrame("last-vision-meta.jpg", meta);

            // Once per draft, keep the WHOLE frame too. The crops alone can't tell you
            // whether a region missed because the ROI is wrong or because the furniture
            // isn't there — only the full frame settles that, and it's what re-calibration
            // needs. Not posted; local diagnostic only.
            if (!_savedFullFrame)
            {
                _savedFullFrame = true;
                Log?.Invoke($"vision: frame {frame.Width}x{frame.Height} — saved last-vision-frame.jpg for ROI calibration");
                var whole = await EncodeFullFrameAsync(frame);
                if (whole != null) SaveFrame("last-vision-frame.jpg", whole);
            }

            var vd = await PostAsync(new (string, byte[])[] { ("teams", teams), ("meta", meta) });

            // If the crops keep coming back with nothing — not even a map — our ROIs are
            // probably wrong for this resolution. Prove it with one full-frame read and
            // latch to that for the rest of the draft rather than staying blind.
            if (vd != null && !vd.FoundNothing) { _emptyReads = 0; return vd; }
            if (++_emptyReads >= EmptyReadsBeforeFallback)
            {
                Log?.Invoke($"vision: {_emptyReads} region reads found nothing — trying a full-frame read (ROI calibration may be off at {frame.Width}x{frame.Height})");
                var full = await RecognizeFullFrameAsync(frame);
                _emptyReads = 0;
                if (full != null && !full.FoundNothing)
                {
                    _fullFrameFallback = true;
                    Log?.Invoke("vision: FULL-FRAME fallback latched for this draft — regions read nothing but the whole frame did");
                    return full;
                }
            }
            return vd;
        }
        catch { return null; }
    }

    // The pre-crop behaviour, kept as a safety net for resolutions our ROIs don't fit.
    private static async System.Threading.Tasks.Task<VisionDraft?> RecognizeFullFrameAsync(ScreenCapture.RawFrame frame)
    {
        var jpeg = await EncodeFullFrameAsync(frame);
        if (jpeg == null) return null;
        SaveFrame("last-vision.jpg", jpeg);
        return await PostAsync(new (string, byte[])[] { ("fullscreen", jpeg) });
    }

    private static async System.Threading.Tasks.Task<VisionDraft?> PostAsync((string Label, byte[] Jpeg)[] regions)
    {
        var payload = JsonSerializer.Serialize(new
        {
            regions = regions.Select(r => new
            {
                label = r.Label,
                imageBase64 = "data:image/jpeg;base64," + Convert.ToBase64String(r.Jpeg),
            }).ToArray(),
        });

        using var req = new HttpRequestMessage(HttpMethod.Post, Endpoint)
        {
            Content = new StringContent(payload, Encoding.UTF8, "application/json"),
        };
        req.Headers.Add("x-vision-token", Token);

        using var resp = await Http.SendAsync(req);
        if (!resp.IsSuccessStatusCode) return null;
        var body = await resp.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<VisionDraft>(body, JsonOpts);
    }

    /// <summary>
    /// One image holding both pick columns side by side, each under a drawn label. The
    /// backend reads our team from the left half and the enemy from the right — a fact
    /// about how we built the image, not a judgement the model has to make.
    /// </summary>
    private static async System.Threading.Tasks.Task<byte[]?> RenderTeamsAsync(
        CanvasDevice device, CanvasBitmap src, int fw, int fh)
    {
        var left = Roi(fw, fh, 0, ColWidthH, ColTopH, ColHeightH, fromRight: false);
        var right = Roi(fw, fh, 0, ColWidthH, ColTopH, ColHeightH, fromRight: true);
        int cw = (int)Math.Max(left.Width, right.Width);
        int ch = (int)Math.Max(left.Height, right.Height);
        if (cw < 8 || ch < 8) return null;

        using var target = new CanvasRenderTarget(device, cw * 2 + Gutter, ch + Header, 96);
        using (var ds = target.CreateDrawingSession())
        {
            ds.Clear(Colors.Black);
            var fmt = new CanvasTextFormat { FontSize = 38 };
            ds.DrawText("OUR TEAM", 10, 10, Colors.White, fmt);
            ds.DrawText("ENEMY TEAM", cw + Gutter + 10, 10, Colors.White, fmt);
            ds.DrawImage(src, new Windows.Foundation.Rect(0, Header, cw, ch), left);
            ds.DrawImage(src, new Windows.Foundation.Rect(cw + Gutter, Header, cw, ch), right);
        }
        return await EncodeAsync(target);
    }

    /// <summary>Bans for both teams plus the map banner, stacked under drawn labels.</summary>
    private static async System.Threading.Tasks.Task<byte[]?> RenderMetaAsync(
        CanvasDevice device, CanvasBitmap src, int fw, int fh)
    {
        var banL = Roi(fw, fh, BanInsetH, BanWidthH, 0, BanHeightH, fromRight: false);
        var banR = Roi(fw, fh, BanInsetH, BanWidthH, 0, BanHeightH, fromRight: true);
        var map = MapRoi(fw, fh);

        int w = (int)Math.Max(Math.Max(banL.Width, banR.Width), map.Width);
        int h = Header * 3 + (int)(banL.Height + banR.Height + map.Height);
        if (w < 8 || h < 8) return null;

        using var target = new CanvasRenderTarget(device, w + 20, h + 10, 96);
        using (var ds = target.CreateDrawingSession())
        {
            ds.Clear(Colors.Black);
            var fmt = new CanvasTextFormat { FontSize = 34 };
            double y = 0;
            void Row(string label, Windows.Foundation.Rect roi)
            {
                ds.DrawText(label, 10, (float)y + 6, Colors.White, fmt);
                y += Header;
                ds.DrawImage(src, new Windows.Foundation.Rect(10, y, roi.Width, roi.Height), roi);
                y += roi.Height;
            }
            Row("MAP", map);
            Row("OUR BANS", banL);
            Row("ENEMY BANS", banR);
        }
        return await EncodeAsync(target);
    }

    // Edge-anchored, height-relative region. Clamped so an unexpected aspect ratio
    // yields a smaller crop rather than an exception.
    private static Windows.Foundation.Rect Roi(
        int fw, int fh, double insetH, double widthH, double topH, double heightH, bool fromRight)
    {
        double w = widthH * fh, h = heightH * fh;
        double y = topH * fh;
        double x = fromRight ? fw - insetH * fh - w : insetH * fh;
        x = Math.Max(0, Math.Min(x, fw - 1));
        y = Math.Max(0, Math.Min(y, fh - 1));
        w = Math.Max(1, Math.Min(w, fw - x));
        h = Math.Max(1, Math.Min(h, fh - y));
        return new Windows.Foundation.Rect(x, y, w, h);
    }

    private static Windows.Foundation.Rect MapRoi(int fw, int fh)
    {
        double w = Math.Min(MapWidthH * fh, fw);
        double h = Math.Min(MapHeightH * fh, fh);
        return new Windows.Foundation.Rect(Math.Max(0, fw / 2.0 - w / 2.0), 0, w, h);
    }

    private static async System.Threading.Tasks.Task<byte[]?> EncodeAsync(CanvasRenderTarget target)
    {
        using var ms = new Windows.Storage.Streams.InMemoryRandomAccessStream();
        await target.SaveAsync(ms, CanvasBitmapFileFormat.Jpeg, 0.9f);
        uint size = (uint)ms.Size;
        var bytes = new byte[size];
        using var reader = new Windows.Storage.Streams.DataReader(ms.GetInputStreamAt(0));
        await reader.LoadAsync(size);
        reader.ReadBytes(bytes);
        return bytes;
    }

    // Keep what we sent on disk so a misread can be inspected against the exact pixels
    // the model saw, rather than guessed at from the log.
    private static void SaveFrame(string name, byte[] jpeg)
    {
        try
        {
            var dir = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HotsFever");
            System.IO.Directory.CreateDirectory(dir);
            System.IO.File.WriteAllBytes(System.IO.Path.Combine(dir, name), jpeg);
        }
        catch { }
    }

    // Downscale the whole BGRA frame and JPEG-encode it (fallback path).
    private static async System.Threading.Tasks.Task<byte[]?> EncodeFullFrameAsync(ScreenCapture.RawFrame f)
    {
        try
        {
            var device = CanvasDevice.GetSharedDevice();
            using var bmp = CanvasBitmap.CreateFromBytes(
                device, f.Bgra, f.Width, f.Height, DirectXPixelFormat.B8G8R8A8UIntNormalized);
            double scale = Math.Min(1.0, (double)TargetWidth / f.Width);
            int tw = Math.Max(1, (int)(f.Width * scale));
            int th = Math.Max(1, (int)(f.Height * scale));
            using var target = new CanvasRenderTarget(device, tw, th, 96);
            using (var ds = target.CreateDrawingSession())
                ds.DrawImage(bmp, new Windows.Foundation.Rect(0, 0, tw, th), bmp.Bounds);
            return await EncodeAsync(target);
        }
        catch { return null; }
    }
}
