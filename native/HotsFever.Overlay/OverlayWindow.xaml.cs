using System.Collections.ObjectModel;
using HotsFever.DraftEngine;
using HotsFever.DraftEngine.Data;
using HotsFever.DraftEngine.Encoding;
using HotsFever.DraftEngine.Models;
using HotsFever.DraftEngine.Rng;
using HotsFever.DraftEngine.Scoring;
using HotsFever.DraftEngine.Search;
using HotsFever.Overlay.Interop;
using Microsoft.UI;
using Microsoft.UI.Composition;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Hosting;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Windows.Graphics;
using Windows.UI;
using WinRT.Interop;

namespace HotsFever.Overlay;

/// <summary>
/// The overlay: a borderless, always-on-top, non-activating live-translucent
/// panel. You tap heroes as they're drafted; each tap advances the draft and
/// re-runs the on-device engine to update recommendations. (Map/tier/team are a
/// fixed demo for now; the setup screen + game-file input are later increments.)
/// </summary>
public sealed partial class OverlayWindow : Window
{
    private IntPtr _hWnd;
    private AppWindow _appWindow = null!;

    private const int PanelWidth = 380;
    private const int ExpandedHeight = 648;
    private const int CollapsedHeight = 112;

    private bool _collapsed;
    private bool _dragging;
    private NativeMethods.POINT _dragStartCursor;
    private PointInt32 _dragStartWin;

    private readonly SolidColorBrush _idleBrush = new(Color.FromArgb(0x99, 0x10, 0x1A, 0x2E));
    private readonly SolidColorBrush _hoverBrush = new(Color.FromArgb(0xF7, 0x10, 0x1A, 0x2E));

    private readonly Microsoft.UI.Dispatching.DispatcherQueueTimer _topmostTimer;

    // ── Draft state ──────────────────────────────────────────────────
    // Fixed 16-step draft order: (team, isPick). Mirrors DRAFT_SEQUENCE.
    private static readonly (int Team, bool IsPick)[] DraftOrder =
    {
        (0, false), (1, false), (0, false), (1, false),
        (0, true),  (1, true),  (1, true),  (0, true),  (0, true),
        (1, false), (0, false),
        (1, true),  (1, true),  (0, true),  (0, true),  (1, true),
    };
    private int _ourTeam = 0;
    private string _map = "Cursed Hollow";
    private string _tier = "mid";
    // Fixed seed so identical draft states give identical (stable) recommendations.
    private const int RngSeed = 1337;

    private bool _draftListCollapsed;
    private bool _setupCollapsed;
    private bool _settingUp = true; // guard combo SelectionChanged during initial population
    private string _dataDir = "";

    private readonly Dictionary<int, string> _selections = new();
    private int _currentStep;

    private OnnxSessions? _sessions;
    private DraftData? _data;
    private PlayerMawpData? _playerData;
    private ReplayScanResult? _scan;
    private string _scanStatus = "scanning replays…";
    private string _lobbyStatus = "";
    private long _lastLobbyWrite;

    public ObservableCollection<RecItem> Recommendations { get; } = new();
    public ObservableCollection<RecItem> YourBest { get; } = new();
    public ObservableCollection<HeroTile> HeroGrid { get; } = new();
    public ObservableCollection<SlotVM> BansMine { get; } = new();
    public ObservableCollection<SlotVM> BansEnemy { get; } = new();
    public ObservableCollection<SlotVM> PicksMine { get; } = new();
    public ObservableCollection<SlotVM> PicksEnemy { get; } = new();

    public OverlayWindow()
    {
        InitializeComponent();
        SystemBackdrop = new WinUIEx.TransparentTintBackdrop();
        ConfigureAsOverlay();

        _topmostTimer = DispatcherQueue.CreateTimer();
        _topmostTimer.Interval = TimeSpan.FromMilliseconds(1000);
        _topmostTimer.Tick += (s, e) => NativeMethods.SetTopMost(_hWnd);
        _topmostTimer.Start();

        RecList.ItemsSource = Recommendations;
        YourBestList.ItemsSource = YourBest;
        HeroGridView.ItemsSource = HeroGrid;
        BansMineList.ItemsSource = BansMine;
        BansEnemyList.ItemsSource = BansEnemy;
        PicksMineList.ItemsSource = PicksMine;
        PicksEnemyList.ItemsSource = PicksEnemy;
        PopulateSetup();
        _ = InitEngineAsync();
        _ = WatchLobbyAsync();
    }

    // ── Engine + draft flow ──────────────────────────────────────────

    private async System.Threading.Tasks.Task InitEngineAsync()
    {
        try
        {
            await System.Threading.Tasks.Task.Run(() =>
            {
                var modelsDir = LocateDir(System.IO.Path.Combine("public", "models"), "draft_policy.onnx");
                _dataDir = LocateDir(System.IO.Path.Combine("src", "lib", "data"), "draft-stats-decayed.json");
                _sessions = OnnxSessions.FromDirectory(modelsDir);
                _data = DraftDataLoader.Load(
                    System.IO.Path.Combine(_dataDir, "draft-stats-decayed.json"),
                    System.IO.Path.Combine(_dataDir, "compositions.json"), _tier);
            });
            PopulateHeroGrid();
            await RecomputeAsync();
            _ = ScanReplaysAsync(); // personalizes (MAWP) once historic replays are parsed
        }
        catch (Exception ex)
        {
            ContextText.Text = "engine error: " + ex.Message;
        }
    }

    // Parse local .StormReplay history into personal stats → MAWP personalization.
    private async System.Threading.Tasks.Task ScanReplaysAsync()
    {
        try
        {
            var replaysDir = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                "Heroes of the Storm", "Accounts");
            if (!System.IO.Directory.Exists(replaysDir)) { _scanStatus = "no Replays folder found"; await RecomputeAsync(); return; }

            var cachePath = ReplayStats.DefaultCachePath();
            void Progress(int d, int t) => DispatcherQueue.TryEnqueue(() => StatusText.Text = $"on-device engine · parsing replays {d}/{t}…");

            var scan = await System.Threading.Tasks.Task.Run(() => ReplayStats.ScanCached(replaysDir, cachePath, Progress));
            _scan = scan;
            _scanStatus = scan.ReplaysParsed > 0
                ? $"personalized from {scan.ReplaysParsed} replays · {scan.LocalBattletag}"
                : "no replays parsed";
            if (scan.LocalBattletag != null && scan.PlayerStats.Count > 0)
            {
                _playerData = new PlayerMawpData
                {
                    PlayerStats = scan.PlayerStats,
                    AvailableBattletags = new[] { scan.LocalBattletag },
                };
            }
            await RecomputeAsync(); // re-run with personalization + updated status
        }
        catch (Exception ex) { _scanStatus = "scan error: " + ex.Message; try { await RecomputeAsync(); } catch { } }
    }

    // Your top heroes by win rate from replay history (10+ games), still available.
    private void UpdateYourBest()
    {
        YourBest.Clear();
        var bt = _scan?.LocalBattletag;
        if (bt == null || _scan == null || !_scan.PlayerStats.TryGetValue(bt, out var byHero))
        {
            YourBestSection.Visibility = Visibility.Collapsed;
            return;
        }
        var taken = new HashSet<string>(_selections.Values, StringComparer.Ordinal);
        var green = new SolidColorBrush(Color.FromArgb(0xFF, 0x4F, 0xFF, 0xB0));
        var red = new SolidColorBrush(Color.FromArgb(0xFF, 0xFF, 0x6B, 0x6B));

        foreach (var (hero, st) in byHero
                     .Where(kv => kv.Value.Games >= 5 && !taken.Contains(kv.Key))
                     .OrderByDescending(kv => kv.Value.Mawp ?? kv.Value.WinRate)
                     .ThenByDescending(kv => kv.Value.Games)
                     .Take(6))
        {
            double metric = st.Mawp ?? st.WinRate; // MAWP = the site's momentum-adjusted %
            YourBest.Add(new RecItem
            {
                Portrait = Short(hero),
                Hero = hero,
                Subtitle = $"win {st.WinRate:0}%  ·  {(int)st.Games} games",
                WinDelta = $"{metric:0}%",
                WinDeltaBrush = metric >= 50 ? green : red,
            });
        }
        YourBestSection.Visibility = YourBest.Count > 0 ? Visibility.Visible : Visibility.Collapsed;
    }

    // Watch for the game's battlelobby at the loading screen; when a fresh one
    // appears, read the real teammates and personalize MAWP for the whole team.
    private async System.Threading.Tasks.Task WatchLobbyAsync()
    {
        var dir = System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Temp", "Heroes of the Storm");
        while (true)
        {
            try
            {
                string? f = System.IO.Directory.Exists(dir)
                    ? System.IO.Directory.EnumerateFiles(dir, "replay.server.battlelobby", System.IO.SearchOption.AllDirectories).FirstOrDefault()
                    : null;
                if (f != null)
                {
                    long wt = System.IO.File.GetLastWriteTimeUtc(f).Ticks;
                    if (wt != _lastLobbyWrite)
                    {
                        _lastLobbyWrite = wt;
                        await System.Threading.Tasks.Task.Delay(1500); // let the write settle (M0 safe-read)
                        var lobby = await System.Threading.Tasks.Task.Run(() => BattlelobbyReader.Read(f));
                        if (lobby != null) ApplyLobby(lobby);
                    }
                }
            }
            catch { }
            await System.Threading.Tasks.Task.Delay(2000);
        }
    }

    private void ApplyLobby(LobbyInfo lobby)
    {
        // Find our team by matching the local player (from replay history) in the lobby.
        var local = _scan?.LocalBattletag;
        int ourTeam = -1;
        if (local != null)
        {
            var me = lobby.Players.FirstOrDefault(p => string.Equals(p.Battletag, local, StringComparison.OrdinalIgnoreCase));
            if (me != null) ourTeam = me.Team;
        }
        if (ourTeam < 0) { _lobbyStatus = $"lobby read ({lobby.Players.Count} players) — you not matched"; return; }

        var teammates = lobby.Players.Where(p => p.Team == ourTeam).Select(p => p.Battletag).ToArray();
        var stats = _playerData?.PlayerStats ?? _scan?.PlayerStats;
        if (stats != null)
        {
            _playerData = new PlayerMawpData { PlayerStats = stats, AvailableBattletags = teammates };
            _lobbyStatus = $"lobby: {teammates.Length} teammates · {lobby.Map}";
            _ = RecomputeAsync(); // personalize for the whole team
        }
    }

    private async System.Threading.Tasks.Task RecomputeAsync()
    {
        UpdateContext();
        BuildDraftBoard();
        UpdateYourBest();
        if (_sessions == null || _data == null || _currentStep >= 16)
        {
            Recommendations.Clear();
            RecSection.Visibility = Visibility.Collapsed;
            return;
        }

        var input = BuildInput();
        bool isBan = !DraftOrder[_currentStep].IsPick;
        var sessions = _sessions;
        var data = _data;
        var playerData = _playerData;

        var result = await System.Threading.Tasks.Task.Run(() =>
            AiInference.GetRecommendations(sessions, input, isBan, new SystemRng(RngSeed),
                new MctsSearch.Options { MinSims = 40, MaxSims = 60, TimeBudgetMs = double.PositiveInfinity },
                data, playerData, topK: 3));

        ApplyRecs(result, isBan);
    }

    private MctsSearch.Input BuildInput()
    {
        var t0 = new List<string>();
        var t1 = new List<string>();
        var bans = new List<string>();
        for (int s = 0; s < _currentStep && s < 16; s++)
        {
            if (!_selections.TryGetValue(s, out var hero)) continue;
            var (team, isPick) = DraftOrder[s];
            if (!isPick) bans.Add(hero);
            else if (team == 0) t0.Add(hero);
            else t1.Add(hero);
        }
        return new MctsSearch.Input
        {
            Team0Picks = t0,
            Team1Picks = t1,
            Bans = bans,
            TakenHeroes = t0.Concat(t1).Concat(bans).ToArray(),
            Map = _map,
            Tier = _tier,
            Step = Math.Min(_currentStep, 15),
            OurTeam = _ourTeam,
        };
    }

    private void UpdateContext()
    {
        if (_currentStep >= 16)
        {
            ContextText.Text = $"{_map}  ·  {Cap(_tier)}";
            RecHeader.Text = "";
            return;
        }
        var (team, isPick) = DraftOrder[_currentStep];
        string who = team == _ourTeam ? "Your" : "Enemy";
        string act = isPick ? "Pick" : "Ban";
        ContextText.Text = $"{who} {act}  ·  {_map}  ·  {Cap(_tier)}  ·  step {_currentStep + 1}/16";
        RecHeader.Text = team == _ourTeam
            ? (isPick ? "RECOMMENDED PICKS" : "RECOMMENDED BANS")
            : (isPick ? "LIKELY ENEMY PICKS" : "LIKELY ENEMY BANS");
    }

    private void ApplyRecs(AiInference.Result result, bool isBan)
    {
        int pct = (int)Math.Round(result.ValueEstimate * 100);
        WinProbText.Text = pct + "%";

        var green = new SolidColorBrush(Color.FromArgb(0xFF, 0x4F, 0xFF, 0xB0));
        var red = new SolidColorBrush(Color.FromArgb(0xFF, 0xFF, 0x6B, 0x6B));

        Recommendations.Clear();
        bool first = true;
        foreach (var rec in result.Recommendations)
        {
            double delta = (rec.WinProb - result.ValueEstimate) * 100.0;
            string playerNote = rec.SuggestedPlayer == null ? ""
                : rec.SuggestedPlayer == _scan?.LocalBattletag ? "  ·  your best"
                : $"  ·  {rec.SuggestedPlayer} should play this";
            Recommendations.Add(new RecItem
            {
                Portrait = Short(rec.Hero),
                Hero = rec.Hero,
                Subtitle = RoleName(rec.Hero) + playerNote,
                WinDelta = (delta >= 0 ? "+" : "") + delta.ToString("0.0") + "%",
                WinDeltaBrush = delta >= 0 ? green : red,
                IsAiPick = first,
            });
            first = false;
        }

        RecSection.Visibility = Recommendations.Count > 0 ? Visibility.Visible : Visibility.Collapsed;
        StatusText.Text = string.IsNullOrEmpty(_lobbyStatus)
            ? $"on-device engine · {result.Sims} sims · {_scanStatus}"
            : $"on-device engine · {result.Sims} sims · {_lobbyStatus}";
    }

    // ── Setup (map / tier / team) ────────────────────────────────────

    private void PopulateSetup()
    {
        MapCombo.ItemsSource = HeroCatalog.Maps;
        MapCombo.SelectedItem = _map;
        TierCombo.ItemsSource = new[] { "Low", "Mid", "High" };
        TierCombo.SelectedItem = Cap(_tier);
        TeamCombo.ItemsSource = new[] { "We ban first (A)", "We ban second (B)" };
        TeamCombo.SelectedIndex = _ourTeam;
        _settingUp = false;
    }

    private void OnMapChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_settingUp || MapCombo.SelectedItem is not string m) return;
        _map = m;
        _ = RecomputeAsync();
    }

    private async void OnTierChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_settingUp || TierCombo.SelectedItem is not string t) return;
        _tier = t.ToLowerInvariant();
        await ReloadDataAsync();
        await RecomputeAsync();
    }

    private void OnTeamChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_settingUp) return;
        _ourTeam = TeamCombo.SelectedIndex < 0 ? 0 : TeamCombo.SelectedIndex;
        _selections.Clear();
        _currentStep = 0;
        if (SearchBox != null) SearchBox.Text = "";
        PopulateHeroGrid();
        _ = RecomputeAsync();
    }

    // Reload the tier-specific stat tables when the tier changes.
    private async System.Threading.Tasks.Task ReloadDataAsync()
    {
        if (string.IsNullOrEmpty(_dataDir)) return;
        var dir = _dataDir; var tier = _tier;
        _data = await System.Threading.Tasks.Task.Run(() => DraftDataLoader.Load(
            System.IO.Path.Combine(dir, "draft-stats-decayed.json"),
            System.IO.Path.Combine(dir, "compositions.json"), tier));
    }

    private void OnToggleSetup(object sender, RoutedEventArgs e)
    {
        _setupCollapsed = !_setupCollapsed;
        SetupBody.Visibility = _setupCollapsed ? Visibility.Collapsed : Visibility.Visible;
        SetupToggle.Content = _setupCollapsed ? "▸" : "▾";
    }

    // ── Hero grid input ──────────────────────────────────────────────

    private void PopulateHeroGrid()
    {
        var taken = new HashSet<string>(_selections.Values, StringComparer.Ordinal);
        string q = (SearchBox?.Text ?? "").Trim();
        HeroGrid.Clear();
        foreach (var h in HeroCatalog.Heroes)
        {
            if (taken.Contains(h)) continue;
            if (q.Length > 0 && h.IndexOf(q, StringComparison.OrdinalIgnoreCase) < 0) continue;
            HeroGrid.Add(new HeroTile { Hero = h, Portrait = Short(h) });
        }
    }

    private void OnHeroClick(object sender, ItemClickEventArgs e)
    {
        if (_currentStep >= 16 || e.ClickedItem is not HeroTile tile) return;
        _selections[_currentStep] = tile.Hero;
        _currentStep++;
        if (SearchBox != null) SearchBox.Text = "";
        PopulateHeroGrid();
        _ = RecomputeAsync();
    }

    private void OnSearchChanged(object sender, TextChangedEventArgs e) => PopulateHeroGrid();

    private void OnResetDraft(object sender, RoutedEventArgs e)
    {
        _selections.Clear();
        _currentStep = 0;
        if (SearchBox != null) SearchBox.Text = "";
        PopulateHeroGrid();
        _ = RecomputeAsync();
    }

    private void OnBack(object sender, RoutedEventArgs e)
    {
        if (_currentStep == 0) return;
        _currentStep--;
        _selections.Remove(_currentStep);
        if (SearchBox != null) SearchBox.Text = "";
        PopulateHeroGrid();
        _ = RecomputeAsync();
    }

    // Mini draft board mirroring the real screen: bans top, our picks left,
    // enemy picks right — each in DRAFT_SEQUENCE order (see buildDraftView).
    private void BuildDraftBoard()
    {
        BansMine.Clear(); BansEnemy.Clear(); PicksMine.Clear(); PicksEnemy.Clear();
        for (int s = 0; s < 16; s++)
        {
            var (team, isPick) = DraftOrder[s];
            _selections.TryGetValue(s, out var hero);
            bool mine = team == _ourTeam;
            var slot = MakeSlot(hero, s == _currentStep, mine);
            if (!isPick) (mine ? BansMine : BansEnemy).Add(slot);
            else (mine ? PicksMine : PicksEnemy).Add(slot);
        }
    }

    private static SlotVM MakeSlot(string? hero, bool isCurrent, bool mine)
    {
        bool filled = hero != null;
        var mineFill = Color.FromArgb(0x40, 0x4A, 0x9E, 0xFF);   // blue
        var enemyFill = Color.FromArgb(0x40, 0xFF, 0x6B, 0x6B);  // red
        var emptyFill = Color.FromArgb(0x12, 0xFF, 0xFF, 0xFF);
        var accent = Color.FromArgb(0xFF, 0x4F, 0xFF, 0xB0);     // current = green

        return new SlotVM
        {
            Portrait = filled ? Short(hero!) : "",
            Name = filled ? hero! : (isCurrent ? "…" : ""),
            Background = new SolidColorBrush(filled ? (mine ? mineFill : enemyFill) : emptyFill),
            BorderColor = new SolidColorBrush(isCurrent ? accent : Color.FromArgb(0, 0, 0, 0)),
            BorderT = new Thickness(isCurrent ? 2 : 0),
            NameForeground = new SolidColorBrush(filled
                ? Color.FromArgb(0xFF, 0xEA, 0xF1, 0xFB)
                : Color.FromArgb(0xFF, 0x7F, 0x93, 0xB0)),
        };
    }

    private void OnToggleDraftList(object sender, RoutedEventArgs e)
    {
        _draftListCollapsed = !_draftListCollapsed;
        DraftPicksBody.Visibility = _draftListCollapsed ? Visibility.Collapsed : Visibility.Visible;
        DraftListToggle.Content = _draftListCollapsed ? "▸" : "▾";
    }

    // Auto-size the window to the panel's content height (so collapsing any
    // section shrinks the window instead of leaving a gap).
    private void OnContentSizeChanged(object sender, SizeChangedEventArgs e)
    {
        if (_appWindow == null || RootGrid.XamlRoot == null) return;
        double scale = RootGrid.XamlRoot.RasterizationScale;
        int target = (int)Math.Ceiling((ContentStack.ActualHeight + 36) * scale); // + Border padding (18*2)
        if (target <= 0 || Math.Abs(target - _appWindow.Size.Height) <= 1) return;
        _appWindow.Resize(new SizeInt32(PanelWidth, target));
        NativeMethods.RoundWindow(_hWnd, _appWindow.Size.Width, _appWindow.Size.Height, 22);
    }

    // ── Labels ───────────────────────────────────────────────────────

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
        // Anywhere on the panel drags — except interactive controls.
        if (IsDescendantOf(e.OriginalSource as DependencyObject, CollapseButton)) return;
        if (IsDescendantOf(e.OriginalSource as DependencyObject, InputArea)) return;
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
        ContentStack.Spacing = _collapsed ? 7 : 12;
        BodyContent.Visibility = _collapsed ? Visibility.Collapsed : Visibility.Visible;
        CollapseButton.Content = _collapsed ? "+" : "–";
        // Window auto-sizes to content via OnContentSizeChanged.
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

        NativeMethods.RoundWindow(hWnd, appWindow.Size.Width, appWindow.Size.Height, 22);
        NativeMethods.AddExStyles(hWnd,
            NativeMethods.WS_EX_TOPMOST | NativeMethods.WS_EX_NOACTIVATE | NativeMethods.WS_EX_TOOLWINDOW);
        NativeMethods.RemoveBorder(hWnd);
        NativeMethods.SetTopMost(hWnd);
    }
}
