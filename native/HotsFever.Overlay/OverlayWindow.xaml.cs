using System.Collections.ObjectModel;
using HotsFever.DraftEngine;
using HotsFever.DraftEngine.Data;
using HotsFever.DraftEngine.Encoding;
using HotsFever.DraftEngine.Models;
using HotsFever.DraftEngine.Rng;
using HotsFever.DraftEngine.Search;
using HotsFever.Overlay.Interop;
using Microsoft.UI;
using Microsoft.UI.Composition;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Hosting;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Windows.Graphics;
using Windows.UI;
using WinRT.Interop;

namespace HotsFever.Overlay;

/// <summary>
/// The overlay: a borderless, always-on-top, non-activating live-translucent
/// panel that runs the on-device draft engine and shows real recommendations.
/// (Demo draft state for now; the hero-grid input + game-file integration are
/// the next M2/M3 increments.)
/// </summary>
public sealed partial class OverlayWindow : Window
{
    private IntPtr _hWnd;
    private AppWindow _appWindow = null!;

    private const int PanelWidth = 380;
    private const int ExpandedHeight = 500;
    private const int CollapsedHeight = 112; // compressed header + context + win-probability

    private bool _collapsed;
    private bool _dragging;
    private NativeMethods.POINT _dragStartCursor;
    private PointInt32 _dragStartWin;

    private readonly SolidColorBrush _idleBrush = new(Color.FromArgb(0x99, 0x10, 0x1A, 0x2E));
    private readonly SolidColorBrush _hoverBrush = new(Color.FromArgb(0xF7, 0x10, 0x1A, 0x2E)); // near-opaque on hover

    private readonly Microsoft.UI.Dispatching.DispatcherQueueTimer _topmostTimer;

    public ObservableCollection<RecItem> Recommendations { get; } = new();

    public OverlayWindow()
    {
        InitializeComponent();
        SystemBackdrop = new WinUIEx.TransparentTintBackdrop();
        ConfigureAsOverlay();

        // Re-assert topmost on a timer so nothing can bury the overlay.
        _topmostTimer = DispatcherQueue.CreateTimer();
        _topmostTimer.Interval = TimeSpan.FromMilliseconds(1000);
        _topmostTimer.Tick += (s, e) => NativeMethods.SetTopMost(_hWnd);
        _topmostTimer.Start();

        RecList.ItemsSource = Recommendations;
        _ = ComputeAsync();
    }

    // ── Engine ───────────────────────────────────────────────────────

    private sealed record EngineResult(AiInference.Result Result, string Map, string Tier);

    private async System.Threading.Tasks.Task ComputeAsync()
    {
        try
        {
            var r = await System.Threading.Tasks.Task.Run(RunEngine);
            DispatcherQueue.TryEnqueue(() => ApplyResult(r));
        }
        catch (Exception ex)
        {
            DispatcherQueue.TryEnqueue(() => ContextText.Text = "engine error: " + ex.Message);
        }
    }

    private EngineResult RunEngine()
    {
        var modelsDir = LocateDir(System.IO.Path.Combine("public", "models"), "draft_policy.onnx");
        var dataDir = LocateDir(System.IO.Path.Combine("src", "lib", "data"), "draft-stats-decayed.json");

        using var sessions = OnnxSessions.FromDirectory(modelsDir);
        var data = DraftDataLoader.Load(
            System.IO.Path.Combine(dataDir, "draft-stats-decayed.json"),
            System.IO.Path.Combine(dataDir, "compositions.json"), "mid");

        // Demo mid-draft state at our first pick (4 bans done).
        var bans = new[] { "Alarak", "Diablo", "Malfurion", "Genji" };
        var input = new MctsSearch.Input
        {
            Bans = bans,
            TakenHeroes = bans,
            Map = "Cursed Hollow",
            Tier = "mid",
            Step = 4,
            OurTeam = 0,
        };

        var result = AiInference.GetRecommendations(sessions, input, isBanStep: false, new SystemRng(),
            new MctsSearch.Options { MinSims = 40, MaxSims = 60, TimeBudgetMs = double.PositiveInfinity },
            data, playerData: null, topK: 5);

        return new EngineResult(result, input.Map, input.Tier);
    }

    private void ApplyResult(EngineResult r)
    {
        ContextText.Text = $"Your Pick  ·  {r.Map}  ·  {Cap(r.Tier)}";

        int pct = (int)Math.Round(r.Result.ValueEstimate * 100);
        WinProbText.Text = pct + "%";

        var green = new SolidColorBrush(Color.FromArgb(0xFF, 0x4F, 0xFF, 0xB0));
        var red = new SolidColorBrush(Color.FromArgb(0xFF, 0xFF, 0x6B, 0x6B));

        Recommendations.Clear();
        bool first = true;
        foreach (var rec in r.Result.Recommendations)
        {
            double delta = (rec.WinProb - r.Result.ValueEstimate) * 100.0;
            string player = rec.SuggestedPlayer != null ? $" · {rec.SuggestedPlayer} should play this" : "";
            Recommendations.Add(new RecItem
            {
                Portrait = Short(rec.Hero),
                Hero = rec.Hero,
                Subtitle = RoleName(rec.Hero) + player,
                WinDelta = (delta >= 0 ? "+" : "") + delta.ToString("0.0") + "%",
                WinDeltaBrush = delta >= 0 ? green : red,
                IsAiPick = first,
            });
            first = false;
        }

        StatusText.Text = $"on-device engine · {r.Result.Sims} sims · demo draft · hero-grid input next";
    }

    // 0-8 fine roles → display labels
    private static readonly string[] RoleNames =
        { "Tank", "Bruiser", "Healer", "Ranged Assassin", "Ranged Assassin", "Melee Assassin", "Support", "Bruiser", "Ranged Assassin" };

    private static string RoleName(string hero)
    {
        int r = EnrichedFeatures.FineRoleOf(hero);
        return r >= 0 ? RoleNames[r] : "";
    }

    private static string Short(string hero)
        => new string(hero.Where(char.IsLetter).Take(3).ToArray());

    private static string Cap(string s)
        => string.IsNullOrEmpty(s) ? s : char.ToUpper(s[0]) + s.Substring(1);

    private static string LocateDir(string relativeDir, string sentinelFile)
    {
        var dir = new System.IO.DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = System.IO.Path.Combine(dir.FullName, relativeDir);
            if (System.IO.File.Exists(System.IO.Path.Combine(candidate, sentinelFile))) return candidate;
            dir = dir.Parent;
        }
        throw new System.IO.DirectoryNotFoundException($"Could not locate {relativeDir} above {AppContext.BaseDirectory}");
    }

    // ── Interactions: hover, drag, collapse ─────────────────────────

    private void OnContentLoaded(object sender, RoutedEventArgs e)
    {
        // Enable implicit Offset animations so every row glides to its new
        // position in sync (same duration) when the layout compresses/expands.
        foreach (var child in ContentStack.Children)
            if (child is UIElement el) EnableImplicitMove(el);
    }

    private static void EnableImplicitMove(UIElement el)
    {
        var visual = ElementCompositionPreview.GetElementVisual(el);
        var compositor = visual.Compositor;
        var offset = compositor.CreateVector3KeyFrameAnimation();
        offset.Target = "Offset";
        offset.InsertExpressionKeyFrame(1.0f, "this.FinalValue");
        offset.Duration = TimeSpan.FromMilliseconds(250);
        var implicitAnims = compositor.CreateImplicitAnimationCollection();
        implicitAnims["Offset"] = offset;
        visual.ImplicitAnimations = implicitAnims;
    }

    private void OnPanelPointerEntered(object sender, PointerRoutedEventArgs e)
        => PanelBorder.Background = _hoverBrush;

    private void OnPanelPointerExited(object sender, PointerRoutedEventArgs e)
    {
        if (!_dragging) PanelBorder.Background = _idleBrush;
    }

    private void OnDragPressed(object sender, PointerRoutedEventArgs e)
    {
        // Anywhere on the panel drags — except the collapse button.
        if (IsDescendantOf(e.OriginalSource as DependencyObject, CollapseButton)) return;
        _dragging = true;
        _dragStartCursor = NativeMethods.CursorPos();
        _dragStartWin = _appWindow.Position;
        RootGrid.CapturePointer(e.Pointer);
    }

    private void OnDragMoved(object sender, PointerRoutedEventArgs e)
    {
        if (!_dragging) return;
        var c = NativeMethods.CursorPos();
        _appWindow.Move(new PointInt32(
            _dragStartWin.X + (c.X - _dragStartCursor.X),
            _dragStartWin.Y + (c.Y - _dragStartCursor.Y)));
    }

    private void OnDragReleased(object sender, PointerRoutedEventArgs e)
    {
        _dragging = false;
        RootGrid.ReleasePointerCapture(e.Pointer);
    }

    private static bool IsDescendantOf(DependencyObject? node, DependencyObject ancestor)
    {
        while (node != null)
        {
            if (node == ancestor) return true;
            node = VisualTreeHelper.GetParent(node);
        }
        return false;
    }

    private void OnToggleCollapse(object sender, RoutedEventArgs e)
    {
        _collapsed = !_collapsed;
        // Compress the remaining rows tighter when collapsed (RepositionThemeTransition
        // on ContentStack glides them to their new Y positions).
        ContentStack.Spacing = _collapsed ? 7 : 12;
        BodyContent.Visibility = _collapsed ? Visibility.Collapsed : Visibility.Visible;
        CollapseButton.Content = _collapsed ? "+" : "–";
        _appWindow.Resize(new SizeInt32(PanelWidth, _collapsed ? CollapsedHeight : ExpandedHeight));
        NativeMethods.RoundWindow(_hWnd, _appWindow.Size.Width, _appWindow.Size.Height, 22);
        NativeMethods.SetTopMost(_hWnd);
    }

    // ── Window styling ───────────────────────────────────────────────

    private void ConfigureAsOverlay()
    {
        var hWnd = WindowNative.GetWindowHandle(this);
        _hWnd = hWnd;
        var windowId = Win32Interop.GetWindowIdFromWindow(hWnd);
        var appWindow = AppWindow.GetFromWindowId(windowId);
        _appWindow = appWindow;

        if (appWindow.Presenter is OverlappedPresenter presenter)
        {
            presenter.SetBorderAndTitleBar(false, false);
            presenter.IsAlwaysOnTop = true;
            presenter.IsResizable = false;
            presenter.IsMaximizable = false;
            presenter.IsMinimizable = false;
        }

        appWindow.Resize(new SizeInt32(PanelWidth, ExpandedHeight));
        appWindow.Move(new PointInt32(48, 48));

        // DWM blur freezes to a stale snapshot on a never-focused window (Win10),
        // so we use a plain translucent fill instead; rounded corners via region.
        NativeMethods.RoundWindow(hWnd, appWindow.Size.Width, appWindow.Size.Height, 22);

        NativeMethods.AddExStyles(hWnd,
            NativeMethods.WS_EX_TOPMOST | NativeMethods.WS_EX_NOACTIVATE | NativeMethods.WS_EX_TOOLWINDOW);
        NativeMethods.RemoveBorder(hWnd);
        NativeMethods.SetTopMost(hWnd);
    }
}
