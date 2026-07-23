using System.Runtime.InteropServices;
using Microsoft.Graphics.Canvas;
using Windows.Graphics.Capture;
using Windows.Graphics.DirectX;
using WinRT;

namespace HotsFever.Overlay;

/// <summary>
/// Windows Graphics Capture (WGC) of a target window — the robust path that
/// works on hardware-accelerated / DirectX surfaces where GDI BitBlt returns
/// black frames. Win2D handles the D3D surface → bitmap → PNG plumbing.
///
/// Increment 1 (M4): prove we can pull real pixels from the HotS window. Later
/// increments crop draft ROIs and template-match hero portraits.
/// </summary>
public static class ScreenCapture
{
    [ComImport]
    [Guid("3628E81B-3CAC-4C60-B7F4-23CE0E0C3356")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IGraphicsCaptureItemInterop
    {
        IntPtr CreateForWindow([In] IntPtr window, [In] ref Guid iid);
        IntPtr CreateForMonitor([In] IntPtr monitor, [In] ref Guid iid);
    }

    private static readonly Guid GraphicsCaptureItemGuid = new("79C3F95B-31F7-4EC2-A464-632EF5D30760");

    private static GraphicsCaptureItem CreateItemForWindow(IntPtr hwnd)
    {
        var interop = GraphicsCaptureItem.As<IGraphicsCaptureItemInterop>();
        Guid iid = GraphicsCaptureItemGuid;
        IntPtr abi = interop.CreateForWindow(hwnd, ref iid);
        var item = GraphicsCaptureItem.FromAbi(abi);
        Marshal.Release(abi);
        return item;
    }

    /// <summary>
    /// Capture a single frame of <paramref name="hwnd"/> as a Win2D bitmap.
    /// Returns null if capture is unsupported or the window is invalid.
    /// Caller owns/disposes the returned bitmap.
    /// </summary>
    public static async System.Threading.Tasks.Task<CanvasBitmap?> CaptureFrameAsync(IntPtr hwnd)
    {
        if (!GraphicsCaptureSession.IsSupported() || hwnd == IntPtr.Zero) return null;

        GraphicsCaptureItem item;
        try { item = CreateItemForWindow(hwnd); }
        catch { return null; }
        if (item == null) return null;

        var device = CanvasDevice.GetSharedDevice();
        var framePool = Direct3D11CaptureFramePool.CreateFreeThreaded(
            device, DirectXPixelFormat.B8G8R8A8UIntNormalized, 2, item.Size);
        var session = framePool.CreateCaptureSession(item);

        var tcs = new System.Threading.Tasks.TaskCompletionSource<CanvasBitmap?>();

        void OnFrameArrived(Direct3D11CaptureFramePool sender, object args)
        {
            try
            {
                using var frame = sender.TryGetNextFrame();
                if (frame == null) return;
                var bmp = CanvasBitmap.CreateFromDirect3D11Surface(device, frame.Surface);
                tcs.TrySetResult(bmp);
            }
            catch (Exception ex) { tcs.TrySetException(ex); }
        }

        framePool.FrameArrived += OnFrameArrived;
        session.StartCapture();

        CanvasBitmap? result = null;
        try
        {
            var completed = await System.Threading.Tasks.Task.WhenAny(
                tcs.Task, System.Threading.Tasks.Task.Delay(3000));
            if (completed == tcs.Task) result = await tcs.Task;
        }
        catch { result = null; }
        finally
        {
            framePool.FrameArrived -= OnFrameArrived;
            session.Dispose();
            framePool.Dispose();
        }
        return result;
    }

    public sealed record RawFrame(byte[] Bgra, int Width, int Height);

    /// <summary>Capture one frame as raw BGRA pixels (for CV), or null on failure.</summary>
    public static async System.Threading.Tasks.Task<RawFrame?> CaptureRawAsync(IntPtr hwnd)
    {
        using var bmp = await CaptureFrameAsync(hwnd);
        if (bmp == null) return null;
        return new RawFrame(bmp.GetPixelBytes(), (int)bmp.SizeInPixels.Width, (int)bmp.SizeInPixels.Height);
    }

    public readonly record struct CaptureInfo(bool Ok, int Width, int Height, double MeanBrightness);

    /// <summary>
    /// Capture the window, save it as a PNG, and report size + mean brightness
    /// (0-255). A near-zero brightness means the capture returned a black frame
    /// (the failure mode WGC is meant to avoid but GDI hits on DirectX surfaces).
    /// </summary>
    public static async System.Threading.Tasks.Task<CaptureInfo> CaptureToPngAsync(IntPtr hwnd, string path)
    {
        using var bmp = await CaptureFrameAsync(hwnd);
        if (bmp == null) return new CaptureInfo(false, 0, 0, 0);

        var file = await EnsureFileAsync(path);
        using (var stream = await file.OpenAsync(Windows.Storage.FileAccessMode.ReadWrite))
            await bmp.SaveAsync(stream, CanvasBitmapFileFormat.Png);

        double mean = 0;
        try
        {
            byte[] px = bmp.GetPixelBytes(); // BGRA
            long sum = 0; long n = 0;
            // sample every ~64th pixel to keep it cheap
            for (int i = 0; i + 2 < px.Length; i += 4 * 64)
            {
                sum += px[i] + px[i + 1] + px[i + 2];
                n += 3;
            }
            mean = n > 0 ? (double)sum / n : 0;
        }
        catch { }

        return new CaptureInfo(true, (int)bmp.SizeInPixels.Width, (int)bmp.SizeInPixels.Height, mean);
    }

    private static async System.Threading.Tasks.Task<Windows.Storage.StorageFile> EnsureFileAsync(string path)
    {
        var dir = System.IO.Path.GetDirectoryName(path)!;
        var name = System.IO.Path.GetFileName(path);
        System.IO.Directory.CreateDirectory(dir);
        var folder = await Windows.Storage.StorageFolder.GetFolderFromPathAsync(dir);
        return await folder.CreateFileAsync(name, Windows.Storage.CreationCollisionOption.ReplaceExisting);
    }
}
