using Microsoft.UI.Xaml;

namespace HotsFever.Overlay;

public partial class App : Application
{
    private Window? _window;

    public App() => InitializeComponent();

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        _window = new OverlayWindow();
        _window.Activate();
    }
}
