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
