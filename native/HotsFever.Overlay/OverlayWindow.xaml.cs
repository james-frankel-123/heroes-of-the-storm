using HotsFever.DraftEngine.Encoding;
using HotsFever.Overlay.Interop;
using Microsoft.UI;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Windows.Graphics;
using WinRT.Interop;

namespace HotsFever.Overlay;

/// <summary>
/// The overlay shell: a borderless, always-on-top, non-activating tool window.
/// (Per-region click-through transparency + the draft board / hero grid are the
/// next M2 increments — this first cut proves the window, the styling, and the
/// engine reference all light up.)
/// </summary>
public sealed partial class OverlayWindow : Window
{
    private IntPtr _hWnd;

    public OverlayWindow()
    {
        InitializeComponent();

        // Make the window itself see-through (WinUIEx's tested transparent
        // backdrop); the frost + opaque text come from the acrylic and XAML.
        SystemBackdrop = new WinUIEx.TransparentTintBackdrop();

        ConfigureAsOverlay();

        // The acrylic / frame / region changes bump z-order after the window
        // activates, so re-assert topmost once the activation settles.
        DispatcherQueue.TryEnqueue(() => NativeMethods.SetTopMost(_hWnd));

        // Footer note (also proves the engine library is wired in).
        StatusText.Text = $"on-device engine · {HeroCatalog.Heroes.Length} heroes · tap a hero to confirm · battletags from game file at load";
    }

    private void ConfigureAsOverlay()
    {
        var hWnd = WindowNative.GetWindowHandle(this);
        _hWnd = hWnd;
        var windowId = Win32Interop.GetWindowIdFromWindow(hWnd);
        var appWindow = AppWindow.GetFromWindowId(windowId);

        if (appWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.SetBorderAndTitleBar(false, false);
            presenter.IsAlwaysOnTop = true;
            presenter.IsResizable = false;
            presenter.IsMaximizable = false;
            presenter.IsMinimizable = false;
        }

        appWindow.Resize(new SizeInt32(380, 500));
        appWindow.Move(new PointInt32(48, 48));

        // NOTE: DWM blur/acrylic (SetWindowCompositionAttribute) freezes to a
        // stale snapshot on a never-focused window (Win10), so it's unsuitable for
        // a live overlay. We use a plain translucent fill (XAML) instead, which
        // DWM composites live. Rounded corners via region.
        NativeMethods.RoundWindow(hWnd, appWindow.Size.Width, appWindow.Size.Height, 22);

        // Stay topmost without stealing focus, and keep out of Alt-Tab.
        // (WS_EX_LAYERED | WS_EX_TRANSPARENT for click-through come with the
        //  per-region hit-testing work in the next increment.)
        NativeMethods.AddExStyles(hWnd,
            NativeMethods.WS_EX_TOPMOST | NativeMethods.WS_EX_NOACTIVATE | NativeMethods.WS_EX_TOOLWINDOW);

        // Strip the leftover 1px system frame that shows as a white outline.
        NativeMethods.RemoveBorder(hWnd);

        // The presenter's IsAlwaysOnTop isn't reliable here — force true topmost.
        NativeMethods.SetTopMost(hWnd);
    }
}
