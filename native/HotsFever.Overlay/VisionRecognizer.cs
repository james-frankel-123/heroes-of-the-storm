using System.Net.Http;
using System.Text;
using System.Text.Json;
using Microsoft.Graphics.Canvas;
using Windows.Graphics.DirectX;

namespace HotsFever.Overlay;

public sealed class VisionDraft
{
    public string? Map { get; set; }
    public List<string> LeftTeam { get; set; } = new();   // the local player's team (always the left column)
    public List<string> RightTeam { get; set; } = new();  // the enemy team
    public List<string> BansLeft { get; set; } = new();
    public List<string> BansRight { get; set; } = new();

    public bool IsEmpty => LeftTeam.Count == 0 && RightTeam.Count == 0 && BansLeft.Count == 0 && BansRight.Count == 0;
}

/// <summary>
/// Reads the live draft screen by sending a downscaled capture to the backend
/// vision endpoint (which runs it through a vision LLM). Robust where OCR and
/// portrait-matching failed — the model understands the image (skins, stylized
/// low-contrast names, patch/UI drift, any resolution).
/// </summary>
public static class VisionRecognizer
{
    private const string Endpoint = "https://hotsfever.com/api/draft-vision";
    // Shared secret matching the Vercel DRAFT_VISION_TOKEN. Gates casual abuse of
    // our vision spend; acceptable to embed for the beta.
    private const string Token = "C6JTbssctxg7ipGxeh6jf83uRIm1Iji3";
    private const int TargetWidth = 1600;

    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(30) };
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    public static async System.Threading.Tasks.Task<VisionDraft?> RecognizeAsync(ScreenCapture.RawFrame frame)
    {
        try
        {
            var jpeg = await EncodeJpegAsync(frame);
            if (jpeg == null) return null;
            var b64 = Convert.ToBase64String(jpeg);
            var payload = JsonSerializer.Serialize(new { imageBase64 = "data:image/jpeg;base64," + b64 });

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
        catch { return null; }
    }

    // Downscale the BGRA frame and JPEG-encode it (via Win2D).
    private static async System.Threading.Tasks.Task<byte[]?> EncodeJpegAsync(ScreenCapture.RawFrame f)
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

            using var ms = new Windows.Storage.Streams.InMemoryRandomAccessStream();
            await target.SaveAsync(ms, CanvasBitmapFileFormat.Jpeg, 0.85f);
            uint size = (uint)ms.Size;
            var bytes = new byte[size];
            using var reader = new Windows.Storage.Streams.DataReader(ms.GetInputStreamAt(0));
            await reader.LoadAsync(size);
            reader.ReadBytes(bytes);
            return bytes;
        }
        catch { return null; }
    }
}
