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
    public OverlayWindow()
    {
        InitializeComponent();
        ConfigureAsOverlay();

        // Prove the engine library is wired in.
        StatusText.Text = $"engine linked — {HeroCatalog.Heroes.Length} heroes, {HeroCatalog.Maps.Length} maps";
    }

    private void ConfigureAsOverlay()
    {
        var hWnd = WindowNative.GetWindowHandle(this);
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

        appWindow.Resize(new SizeInt32(340, 150));
        appWindow.Move(new PointInt32(48, 48));

        // Stay topmost without stealing focus, and keep out of Alt-Tab.
        // (WS_EX_LAYERED | WS_EX_TRANSPARENT for click-through come with the
        //  per-region hit-testing work in the next increment.)
        NativeMethods.AddExStyles(hWnd,
            NativeMethods.WS_EX_TOPMOST | NativeMethods.WS_EX_NOACTIVATE | NativeMethods.WS_EX_TOOLWINDOW);

        // The presenter's IsAlwaysOnTop isn't reliable here — force true topmost.
        NativeMethods.SetTopMost(hWnd);
    }
}
