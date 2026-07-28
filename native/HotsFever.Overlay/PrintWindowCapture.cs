using System.Runtime.InteropServices;

namespace HotsFever.Overlay;

/// <summary>
/// Border-free window capture via GDI PrintWindow(PW_RENDERFULLCONTENT). Unlike
/// Windows Graphics Capture, this draws no yellow capture border (which Windows 10
/// can't hide) — verified to read the HotS draft screen at full fidelity. Returns
/// BGRA pixels (top-down), the format DraftDetector expects.
/// </summary>
public static class PrintWindowCapture
{
    [DllImport("user32.dll")] private static extern IntPtr GetWindowDC(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern int ReleaseDC(IntPtr hWnd, IntPtr hDC);
    [DllImport("user32.dll")] private static extern bool GetWindowRect(IntPtr hWnd, out RECT r);
    [DllImport("user32.dll")] private static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, uint flags);
    [DllImport("gdi32.dll")] private static extern IntPtr CreateCompatibleDC(IntPtr hdc);
    [DllImport("gdi32.dll")] private static extern bool DeleteDC(IntPtr hdc);
    [DllImport("gdi32.dll")] private static extern IntPtr SelectObject(IntPtr hdc, IntPtr h);
    [DllImport("gdi32.dll")] private static extern bool DeleteObject(IntPtr h);
    [DllImport("gdi32.dll")] private static extern IntPtr CreateDIBSection(IntPtr hdc, ref BITMAPINFO bmi, uint usage, out IntPtr bits, IntPtr section, uint offset);

    [StructLayout(LayoutKind.Sequential)] private struct RECT { public int Left, Top, Right, Bottom; }

    [StructLayout(LayoutKind.Sequential)]
    private struct BITMAPINFOHEADER
    {
        public uint biSize;
        public int biWidth, biHeight;
        public ushort biPlanes, biBitCount;
        public uint biCompression, biSizeImage;
        public int biXPelsPerMeter, biYPelsPerMeter;
        public uint biClrUsed, biClrImportant;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BITMAPINFO { public BITMAPINFOHEADER bmiHeader; public uint bmiColors; }

    private const uint PW_RENDERFULLCONTENT = 2;
    private const uint BI_RGB = 0;
    private const uint DIB_RGB_COLORS = 0;

    /// <summary>Capture the window's client-area pixels (BGRA), or null on failure.</summary>
    public static ScreenCapture.RawFrame? Capture(IntPtr hwnd)
    {
        if (hwnd == IntPtr.Zero || !GetWindowRect(hwnd, out var r)) return null;
        int w = r.Right - r.Left, h = r.Bottom - r.Top;
        if (w <= 0 || h <= 0) return null;

        IntPtr winDC = GetWindowDC(hwnd);
        IntPtr memDC = CreateCompatibleDC(winDC);
        var bmi = new BITMAPINFO();
        bmi.bmiHeader.biSize = (uint)Marshal.SizeOf<BITMAPINFOHEADER>();
        bmi.bmiHeader.biWidth = w;
        bmi.bmiHeader.biHeight = -h; // negative = top-down
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        IntPtr dib = CreateDIBSection(memDC, ref bmi, DIB_RGB_COLORS, out IntPtr bits, IntPtr.Zero, 0);
        IntPtr old = SelectObject(memDC, dib);

        byte[]? buf = null;
        try
        {
            if (PrintWindow(hwnd, memDC, PW_RENDERFULLCONTENT) && bits != IntPtr.Zero)
            {
                buf = new byte[w * h * 4];
                Marshal.Copy(bits, buf, 0, buf.Length);
            }
        }
        finally
        {
            SelectObject(memDC, old);
            DeleteObject(dib);
            DeleteDC(memDC);
            ReleaseDC(hwnd, winDC);
        }
        return buf == null ? null : new ScreenCapture.RawFrame(buf, w, h);
    }
}
