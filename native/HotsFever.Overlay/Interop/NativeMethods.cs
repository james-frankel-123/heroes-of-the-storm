using System.Runtime.InteropServices;

namespace HotsFever.Overlay.Interop;

/// <summary>
/// Win32 window-style interop for the overlay. The extended styles let the window
/// stay topmost without stealing focus (NOACTIVATE), hide from Alt-Tab
/// (TOOLWINDOW), and — eventually — become click-through per-region (LAYERED +
/// TRANSPARENT), which is toggled on the transparent background but off over the
/// interactive panel.
/// </summary>
internal static class NativeMethods
{
    public const int GWL_EXSTYLE = -20;
    public const int GWL_STYLE = -16;

    private const long WS_BORDER = 0x00800000;
    private const long WS_DLGFRAME = 0x00400000;
    private const long WS_THICKFRAME = 0x00040000;
    private const long WS_CAPTION = 0x00C00000; // WS_BORDER | WS_DLGFRAME
    private const uint SWP_NOZORDER = 0x0004;
    private const uint SWP_FRAMECHANGED = 0x0020;

    public const long WS_EX_TOPMOST = 0x00000008;
    public const long WS_EX_TRANSPARENT = 0x00000020;
    public const long WS_EX_TOOLWINDOW = 0x00000080;
    public const long WS_EX_LAYERED = 0x00080000;
    public const long WS_EX_NOACTIVATE = 0x08000000;

    private static readonly IntPtr HWND_TOPMOST = new(-1);
    private const uint SWP_NOSIZE = 0x0001;
    private const uint SWP_NOMOVE = 0x0002;
    private const uint SWP_NOACTIVATE = 0x0010;
    private const uint SWP_SHOWWINDOW = 0x0040;

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter,
        int X, int Y, int cx, int cy, uint uFlags);

    /// <summary>Force the window into the topmost band without activating it.</summary>
    public static void SetTopMost(IntPtr hWnd)
        => SetWindowPos(hWnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);

    [StructLayout(LayoutKind.Sequential)]
    private struct AccentPolicy
    {
        public int AccentState;
        public int AccentFlags;
        public uint GradientColor;
        public int AnimationId;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct WindowCompositionAttributeData
    {
        public int Attribute;
        public IntPtr Data;
        public int SizeOfData;
    }

    [DllImport("user32.dll")]
    private static extern int SetWindowCompositionAttribute(IntPtr hwnd, ref WindowCompositionAttributeData data);

    private const int WCA_ACCENT_POLICY = 19;
    private const int ACCENT_ENABLE_BLURBEHIND = 3;        // lighter, more transparent glass
    private const int ACCENT_ENABLE_ACRYLICBLURBEHIND = 4; // frostier, but a higher opacity floor

    /// <summary>
    /// Glass blur behind the window via DWM — works regardless of focus (unlike
    /// WinUI's DesktopAcrylicBackdrop). `acrylic` = frostier but less transparent;
    /// plain blur is more see-through. Tint is 0xAABBGGRR (ABGR).
    /// </summary>
    public static void EnableAcrylic(IntPtr hWnd, uint gradientColorAbgr, bool acrylic = false)
    {
        var accent = new AccentPolicy
        {
            AccentState = acrylic ? ACCENT_ENABLE_ACRYLICBLURBEHIND : ACCENT_ENABLE_BLURBEHIND,
            AccentFlags = 0,
            GradientColor = gradientColorAbgr,
            AnimationId = 0,
        };
        int size = Marshal.SizeOf(accent);
        IntPtr ptr = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(accent, ptr, false);
            var data = new WindowCompositionAttributeData
            {
                Attribute = WCA_ACCENT_POLICY,
                Data = ptr,
                SizeOfData = size,
            };
            SetWindowCompositionAttribute(hWnd, ref data);
        }
        finally { Marshal.FreeHGlobal(ptr); }
    }

    [DllImport("gdi32.dll")]
    private static extern IntPtr CreateRoundRectRgn(int left, int top, int right, int bottom, int widthEllipse, int heightEllipse);

    [DllImport("user32.dll")]
    private static extern int SetWindowRgn(IntPtr hWnd, IntPtr hRgn, bool bRedraw);

    /// <summary>Clip the window to a rounded rectangle (rounded corners on Win10, where DWM won't do it).</summary>
    public static void RoundWindow(IntPtr hWnd, int width, int height, int radius)
    {
        var rgn = CreateRoundRectRgn(0, 0, width + 1, height + 1, radius, radius);
        SetWindowRgn(hWnd, rgn, true);
    }

    /// <summary>Strip the window's non-client frame (the 1px system border WinUI leaves on a "borderless" window).</summary>
    public static void RemoveBorder(IntPtr hWnd)
    {
        long style = GetWindowLongPtr64(hWnd, GWL_STYLE).ToInt64();
        style &= ~(WS_CAPTION | WS_THICKFRAME | WS_BORDER | WS_DLGFRAME);
        SetWindowLongPtr64(hWnd, GWL_STYLE, new IntPtr(style));
        SetWindowPos(hWnd, IntPtr.Zero, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
    }

    [DllImport("user32.dll", SetLastError = true, EntryPoint = "GetWindowLongPtrW")]
    private static extern IntPtr GetWindowLongPtr64(IntPtr hWnd, int nIndex);

    [DllImport("user32.dll", SetLastError = true, EntryPoint = "SetWindowLongPtrW")]
    private static extern IntPtr SetWindowLongPtr64(IntPtr hWnd, int nIndex, IntPtr dwNewLong);

    public static long GetExStyle(IntPtr hWnd) => GetWindowLongPtr64(hWnd, GWL_EXSTYLE).ToInt64();

    public static void SetExStyle(IntPtr hWnd, long exStyle)
        => SetWindowLongPtr64(hWnd, GWL_EXSTYLE, new IntPtr(exStyle));

    /// <summary>Add extended styles to a window.</summary>
    public static void AddExStyles(IntPtr hWnd, long styles)
        => SetExStyle(hWnd, GetExStyle(hWnd) | styles);
}
