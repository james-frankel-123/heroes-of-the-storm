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
    private OverlappedPresenter? _presenter;

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

    // Vision-draft accumulation, rebuilt into _selections each read and cleared by
    // ResetForNewGame. A hero reaches the board only after ConfirmReads consecutive
    // reads in the SAME slot (rejects transient mislabels and L/R swaps), and comes
    // back off after RevokeReads consecutive reads missing. Revocation is what stops
    // a hovered/previewed pick from sticking: a locked portrait never disappears, so
    // only an abandoned preview or a bad read ever ages out.
    private readonly List<string> _vOurPicks = new();
    private readonly List<string> _vEnemyPicks = new();
    private readonly List<string> _vOurBans = new();
    private readonly List<string> _vEnemyBans = new();
    private readonly Dictionary<string, (int side, bool pick)> _vCommitted = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ((int side, bool pick) slot, int hits)> _vCandidates = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _vMisses = new(StringComparer.Ordinal);
    // Display-only: step -> hero a player has SHOWN but not locked. Never in _selections.
    private readonly Dictionary<int, string> _vShowing = new();
    // 2, not 3. Confirmation is latency you pay on EVERY pick — at ~3s a read, a third
    // confirming read pushed a hero onto the board ~9s after it was locked on screen,
    // which is most of the delay you feel. 3 was set when the model was being handed the
    // whole screen and inventing heroes from the centre splash; on the column crops a
    // repeated same-slot misread is far rarer, and RevokeReads still ages out the ones
    // that slip through.
    private const int ConfirmReads = 2; // reads in one slot before it lands
    private const int RevokeReads = 3;  // reads missing before it leaves again

    // The 16-step order is fixed, so only a handful of (leftPicks, rightPicks) counts
    // are ever reachable. The vision model has no such constraint and routinely reports
    // impossible boards (e.g. 5 picks for us against 2 for them, sustained for 35s), so
    // this rejects those reads outright. Which column drafts first varies by game, so
    // both orientations are accepted.
    private static readonly HashSet<(int Left, int Right)> ReachablePickCounts = BuildReachablePickCounts();

    private static HashSet<(int, int)> BuildReachablePickCounts()
    {
        var set = new HashSet<(int, int)> { (0, 0) };
        int a = 0, b = 0;
        foreach (var (team, isPick) in DraftOrder)
        {
            if (!isPick) continue;
            if (team == 0) a++; else b++;
            set.Add((a, b));
            set.Add((b, a));
        }
        return set;
    }

    private OnnxSessions? _sessions;
    private DraftData? _data;
    private PlayerMawpData? _playerData;
    private ReplayScanResult? _scan;
    private string _scanStatus = "scanning replays…";
    private string _lobbyStatus = "";
    private long _lastLobbyWrite;

    public ObservableCollection<RecItem> Recommendations { get; } = new();
    public ObservableCollection<ThreatItem> Threats { get; } = new();

    // Distribution/health notices. _visionDown is set after a run of failed reads so a
    // dead endpoint degrades to manual mode with an explanation instead of silence.
    private string? _updateNotice;
    private bool _updateDismissed;
    private bool _visionDown;
    private bool _visionDownDismissed;
    private int _visionFailures;
    private const int VisionFailuresBeforeNotice = 4;
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
        _topmostTimer.Interval = TimeSpan.FromMilliseconds(400);
        _topmostTimer.Tick += (s, e) => ReassertTopMost();
        _topmostTimer.Start();

        RecList.ItemsSource = Recommendations;
        ThreatList.ItemsSource = Threats;
        YourBestList.ItemsSource = YourBest;
        HeroGridView.ItemsSource = HeroGrid;
        BansMineList.ItemsSource = BansMine;
        BansEnemyList.ItemsSource = BansEnemy;
        PicksMineList.ItemsSource = PicksMine;
        PicksEnemyList.ItemsSource = PicksEnemy;
        PopulateSetup();
        VersionText.Text = "v" + AppVersion.Current;
        DraftLog($"HotS Fever Draft Coach v{AppVersion.Current} starting");
        _ = InitEngineAsync();
        _ = WatchLobbyAsync();
        _ = DraftWatchAsync();
        _ = CheckForUpdateAsync();

        _welcomePending = IsFirstRun();
        UpdateNotice();
    }

    // ── Engine + draft flow ──────────────────────────────────────────

    private async System.Threading.Tasks.Task InitEngineAsync()
    {
        try
        {
            await System.Threading.Tasks.Task.Run(() =>
            {
                var modelsDir = ResolveAssetDir("models", "draft_policy.onnx", System.IO.Path.Combine("public", "models"));
                _dataDir = ResolveAssetDir("data", "draft-stats-decayed.json", System.IO.Path.Combine("src", "lib", "data"));
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
    private bool _inDraft;
    private bool _isFullscreen;    // HotS in exclusive fullscreen (overlay can't show)
    private bool _fsDismissed;     // fullscreen warning dismissed this episode
    private bool _welcomePending;  // first-run onboarding not yet dismissed

    private static readonly string FirstRunMarker = System.IO.Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HotsFever", "firstrun.done");

    private static bool IsFirstRun() => !System.IO.File.Exists(FirstRunMarker);

    private static void MarkFirstRunDone()
    {
        try
        {
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(FirstRunMarker)!);
            System.IO.File.WriteAllText(FirstRunMarker, DateTime.Now.ToString("o"));
        }
        catch { }
    }

    // The above-the-fold notice banner shows the fullscreen warning (top priority,
    // actionable) or the first-run onboarding tip. Must be called on the UI thread.
    private void UpdateNotice()
    {
        if (_isFullscreen && !_fsDismissed)
        {
            NoticeText.Text = "⚠ Heroes of the Storm is in EXCLUSIVE FULLSCREEN — the overlay can't draw over it. " +
                              "Switch Options → Video → Display Mode to \"Borderless Windowed\".";
            NoticeBanner.Visibility = Visibility.Visible;
        }
        else if (_welcomePending)
        {
            NoticeText.Text = "Welcome to HotS Fever! Set HotS → Options → Video → Display Mode to " +
                              "\"Borderless Windowed\" so the overlay shows over the game. " +
                              "Recommendations update live during your draft.";
            NoticeBanner.Visibility = Visibility.Visible;
        }
        else if (_visionDown && !_visionDownDismissed)
        {
            // Recognition needs our endpoint; without it the overlay still works, but
            // only from your taps. Say so rather than looking silently broken.
            NoticeText.Text = "⚠ Can't reach the draft-reading service, so the board won't fill in " +
                              "automatically. Everything else still works — tap heroes in the grid " +
                              "to drive the draft by hand.";
            NoticeBanner.Visibility = Visibility.Visible;
        }
        else if (_updateNotice != null && !_updateDismissed)
        {
            NoticeText.Text = "⬆ " + _updateNotice;
            NoticeBanner.Visibility = Visibility.Visible;
        }
        else NoticeBanner.Visibility = Visibility.Collapsed;
    }

    // There is no auto-updater: installs are per-user and unattended replacement would
    // be a bigger promise than this app should make. Telling you a new build exists is
    // the useful 90% — the banner links to the download and the installer upgrades in
    // place over the same AppId.
    private async System.Threading.Tasks.Task CheckForUpdateAsync()
    {
        await System.Threading.Tasks.Task.Delay(4000); // let startup settle first
        string? notice = await AppVersion.CheckForUpdateAsync();
        if (notice == null) return;
        DraftLog("update available: " + notice);
        DispatcherQueue.TryEnqueue(() => { _updateNotice = notice; UpdateNotice(); });
    }

    private void OnDismissNotice(object sender, RoutedEventArgs e)
    {
        if (_isFullscreen && !_fsDismissed) _fsDismissed = true;
        else if (_welcomePending) { _welcomePending = false; MarkFirstRunDone(); }
        else if (_visionDown && !_visionDownDismissed) _visionDownDismissed = true;
        else if (_updateNotice != null) _updateDismissed = true;
        UpdateNotice();
    }

    // CV screen-capture uses border-free PrintWindow (no Win10 yellow capture
    // border, unlike WGC) and is scoped to the draft: it captures during the
    // pick/ban phase and HARD-STOPS the moment the battlelobby file appears (draft
    // complete → loading), staying stopped through the match. Per-hero recognition
    // (auto-filling picks/bans) plugs into the marked spot below — until that
    // lands, capture only detects the draft; the board still fills from the
    // battlelobby file at loading and live recs come from manual taps.
    private const bool EnableCvCapture = true;
    // Auto-fill life-cycle. The battlelobby file lingers after the match on
    // machines without the HeroesProfile uploader cleaning it, so its existence
    // is NOT a reliable match-end signal. Instead, once the board is auto-filled
    // from a completed lobby, we treat the match as over when the game stops
    // WRITING its in-progress replay (TempWriteReplayP1) for a sustained window.
    private bool _autoFilled;                        // board came from a completed battlelobby fill
    private const double MatchStaleSeconds = 120;    // temp replay writes older than this ⇒ no live match
    private long _draftEndedAtMs;                    // TickCount64 when the detector last lost the draft
    private const double ResumeGraceSeconds = 150;   // re-detect within this ⇒ same draft, keep the board

    private async System.Threading.Tasks.Task DraftWatchAsync()
    {
        await System.Threading.Tasks.Task.Delay(2500);
        var dir = System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HotsFever");
        // The game creates this dir at the loading screen and cleans it after the
        // match — so its presence marks "loading/in-match", when we must NOT capture.
        var battlelobbyDir = System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Temp", "Heroes of the Storm");
        DraftLog($"HotS watcher started (cv capture {(EnableCvCapture ? "ON, draft-gated" : "OFF")})");
        VisionRecognizer.Log = DraftLog; // so ROI/fallback problems land in the same log

        int tick = 0;
        int lowStreak = 0;
        while (true)
        {
            int delay = 3000;
            try
            {
                var hots = System.Diagnostics.Process.GetProcessesByName("HeroesOfTheStorm_x64").FirstOrDefault();
                IntPtr hwnd = hots != null ? NativeMethods.FindWindowForProcess(hots.Id) : IntPtr.Zero;

                if (hwnd == IntPtr.Zero)
                {
                    if (_inDraft) _inDraft = false;
                    if (_isFullscreen) { _isFullscreen = false; _fsDismissed = false; UpdateNotice(); }
                }
                else
                {
                    // Cheap: warn if HotS is in exclusive fullscreen (overlay can't show over it).
                    bool fs = NativeMethods.IsExclusiveFullscreen();
                    if (fs != _isFullscreen)
                    {
                        _isFullscreen = fs;
                        if (!fs) _fsDismissed = false; // re-warn on a new fullscreen episode
                        UpdateNotice();
                    }

                    // Gate purely on the draft-screen detector (score ~1.0 in draft, ~0.5
                    // otherwise). The battlelobby-dir latch was unreliable — that folder
                    // lingers into the next game and lingered-blocked the new draft.
                    if (EnableCvCapture)
                    {
                        var raw = await System.Threading.Tasks.Task.Run(() => PrintWindowCapture.Capture(hwnd));
                        if (raw != null)
                        {
                            double score = await System.Threading.Tasks.Task.Run(
                                () => DraftDetector.Score(raw.Bgra, raw.Width, raw.Height));
                            // Hysteresis + confirmation. Real draft frames score ~1.0 while
                            // in-game/loading noise peaks ~0.65, so START only on a clearly
                            // high score, and END only after several CONSECUTIVE low frames.
                            // A single weak/animation frame (e.g. a lock-in transition) no
                            // longer flaps the board out and resets it.
                            const double StartThreshold = 0.85;
                            const double EndThreshold = 0.75;
                            const int EndConfirmFrames = 3;
                            if (score < EndThreshold) lowStreak++; else lowStreak = 0;
                            if (++tick % 6 == 0) DraftLog($"poll: score {score:F2}, inDraft={_inDraft}, low={lowStreak}");
                            if (!_inDraft && score >= StartThreshold)
                            {
                                _inDraft = true; lowStreak = 0;
                                // The template matches live frames weakly, so it can lose a draft
                                // that's still running (seen: a 53s dropout mid-draft). Treat a
                                // quick re-detect as the SAME draft and keep the board; only a
                                // long gap means a genuinely new one.
                                double gap = (Environment.TickCount64 - _draftEndedAtMs) / 1000.0;
                                bool resumed = _draftEndedAtMs > 0 && gap < ResumeGraceSeconds;
                                DraftLog($"draft STARTED ({score:F2}){(resumed ? $" — resumed after {gap:F0}s gap, keeping board" : "")}");
                                if (!resumed) { ResetForNewGame(); VisionRecognizer.ResetForNewDraft(); }
                            }
                            else if (_inDraft && lowStreak >= EndConfirmFrames)
                            {
                                _inDraft = false;
                                _draftEndedAtMs = Environment.TickCount64;
                                // Don't wipe the board here — this fires on detector dropouts too.
                                // A real new draft resets on start; a finished match clears via
                                // the lobby watcher; stale heroes otherwise age out via RevokeReads.
                                DraftLog($"draft ended (sustained low, last {score:F2})");
                            }
                            if (_inDraft)
                            {
                                // The vision round-trip (~3s) already paces this loop; the
                                // extra sleep was pure added latency on top of it. Keep a
                                // small gap so a failed/instant call can't spin the CPU.
                                delay = 250;
                                var vd = await VisionRecognizer.RecognizeAsync(raw);
                                // A null read means the request itself failed (offline, endpoint
                                // down, token rejected) — not that the screen was empty.
                                if (vd == null)
                                {
                                    if (++_visionFailures >= VisionFailuresBeforeNotice && !_visionDown)
                                    {
                                        _visionDown = true;
                                        DraftLog($"vision unreachable after {_visionFailures} attempts — falling back to manual entry");
                                        UpdateNotice();
                                    }
                                }
                                else if (_visionFailures > 0)
                                {
                                    _visionFailures = 0;
                                    if (_visionDown) { _visionDown = false; _visionDownDismissed = false; UpdateNotice(); }
                                }
                                if (vd != null && !vd.IsEmpty)
                                {
                                    DraftLog($"vision: L[{string.Join(",", vd.LeftTeam)}] R[{string.Join(",", vd.RightTeam)}] " +
                                             $"bans[{string.Join(",", vd.BansLeft.Concat(vd.BansRight))}] " +
                                             $"pending[{string.Join(",", vd.PendingLeft.Concat(vd.PendingRight))}] " +
                                             $"preview={vd.PreviewHero ?? "-"} map={vd.Map} " +
                                             $"[{vd.Model ?? "?"} {vd.PromptTokens}+{vd.CompletionTokens} tok]");
                                    await ApplyVisionDraft(vd);
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex) { DraftLog("watch error: " + ex.Message); }
            await System.Threading.Tasks.Task.Delay(delay);
        }
    }

    private static readonly string DraftLogPath = System.IO.Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HotsFever", "draft-watch.log");

    private static void DraftLog(string msg)
    {
        try
        {
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(DraftLogPath)!);
            System.IO.File.AppendAllText(DraftLogPath, $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}  {msg}{Environment.NewLine}");
        }
        catch { }
    }

    private static readonly string LobbyLogPath = System.IO.Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HotsFever", "lobby-watch.log");

    private static void LobbyLog(string msg)
    {
        try
        {
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(LobbyLogPath)!);
            System.IO.File.AppendAllText(LobbyLogPath, $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}  {msg}{Environment.NewLine}");
        }
        catch { }
    }

    private async System.Threading.Tasks.Task WatchLobbyAsync()
    {
        // The game writes replay.server.battlelobby under a TempWriteReplayP1 subfolder
        // at the loading screen (M0), then it's cleaned up shortly after the match — so
        // only a live loading screen produces it. The log makes each detection auditable.
        var dir = System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Temp", "Heroes of the Storm");
        LobbyLog($"watcher started — watching {dir}");
        var opts = new System.IO.EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true };
        bool dirSeen = false;
        while (true)
        {
            try
            {
                bool exists = System.IO.Directory.Exists(dir);
                if (exists && !dirSeen) { dirSeen = true; LobbyLog("lobby dir appeared (loading screen)"); }
                else if (!exists && dirSeen)
                {
                    // Match ended and the game cleaned up its lobby file — clear the
                    // auto-filled draft so we don't show a stale board into the next game.
                    dirSeen = false;
                    _lastLobbyWrite = 0; // next lobby always counts as fresh
                    LobbyLog("lobby dir gone (match ended) — clearing stale draft");
                    ResetForNewGame();
                }

                string? f = exists
                    ? System.IO.Directory.EnumerateFiles(dir, "replay.server.battlelobby", opts).FirstOrDefault()
                    : null;
                if (f != null)
                {
                    long wt = System.IO.File.GetLastWriteTimeUtc(f).Ticks;
                    if (wt != _lastLobbyWrite)
                    {
                        _lastLobbyWrite = wt;
                        LobbyLog($"fresh battlelobby found: {f}");
                        await System.Threading.Tasks.Task.Delay(1500); // let the write settle (M0 safe-read)
                        var lobby = await System.Threading.Tasks.Task.Run(() => BattlelobbyReader.Read(f));
                        if (lobby == null)
                            LobbyLog("decode returned null (HeroesDecode missing or parse failed)");
                        else if (!MatchLikelyActive())
                            // The battlelobby file lingers after a match on machines without the
                            // HeroesProfile uploader; on startup we'd otherwise re-fill a finished
                            // game. Only auto-fill when the game is actually writing its replay.
                            LobbyLog($"battlelobby found but temp replay writes are stale — skipping stale auto-fill (map {lobby.Map})");
                        else
                        {
                            LobbyLog($"decoded {lobby.Players.Count} players · map {lobby.Map} · mode {lobby.GameMode}");
                            ApplyLobby(lobby);
                        }
                    }
                }

                // Clear a stale auto-filled board once the match is over. The game
                // writes its in-progress replay under TempWriteReplayP1 all through
                // loading and gameplay; when the newest write is older than the stale
                // window we're back at the menu, so the board should reset. (The lobby
                // dir/file itself is an unreliable signal — it lingers after the game.)
                if (_autoFilled && !_inDraft && !MatchLikelyActive())
                {
                    LobbyLog("match over (temp replay writes stale) — clearing stale auto-filled board");
                    ResetForNewGame();
                }
            }
            catch (Exception ex) { LobbyLog("watch error: " + ex.Message); }
            await System.Threading.Tasks.Task.Delay(2000);
        }
    }

    // True while the game is actively writing its in-progress replay (loading or
    // in a match): the newest TempWriteReplayP1 write is recent. False at the menu,
    // where those writes go stale — and false if the folder is absent. This is the
    // signal for "a live match is happening", independent of the lingering lobby file.
    private static bool MatchLikelyActive()
    {
        try
        {
            var dir = System.IO.Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Temp", "Heroes of the Storm", "TempWriteReplayP1");
            if (!System.IO.Directory.Exists(dir)) return false;
            long newest = 0;
            foreach (var f in System.IO.Directory.EnumerateFiles(dir))
            {
                long t = System.IO.File.GetLastWriteTimeUtc(f).Ticks;
                if (t > newest) newest = t;
            }
            if (newest == 0) return false;
            return (DateTime.UtcNow - new DateTime(newest, DateTimeKind.Utc)).TotalSeconds < MatchStaleSeconds;
        }
        catch { return false; }
    }

    private void ApplyLobby(LobbyInfo lobby)
    {
        // Find our team. Primary: match a lobby player's toon id to a local account
        // folder (definitive even with multiple accounts). Fallback: the scan's battletag.
        int ourTeam = -1;
        LobbyPlayer? me = null;
        var localToons = BattlelobbyReader.GetLocalToonIds();
        me = lobby.Players.FirstOrDefault(p => localToons.Contains(p.ToonId));
        if (me == null && _scan?.LocalBattletag is string local)
            me = lobby.Players.FirstOrDefault(p => string.Equals(p.Battletag, local, StringComparison.OrdinalIgnoreCase));
        if (me != null) ourTeam = me.Team;

        if (ourTeam < 0)
        {
            _lobbyStatus = $"lobby read ({lobby.Players.Count} players) — you not matched";
            LobbyLog($"applied but no local account matched. lobby toons: {string.Join(", ", lobby.Players.Select(p => p.ToonId))} | local toons: {string.Join(", ", localToons)}");
            return;
        }

        // Personalize "your best" to the account actually in this game.
        if (_scan != null && me!.Battletag.Length > 0) _scan.LocalBattletag = me.Battletag;

        // Auto-fill the board with the real final draft (picks + bans) from the lobby.
        FillDraftFromLobby(lobby, ourTeam);
        _autoFilled = true; // watched for match-over so the board doesn't go stale (see below)

        var teammates = lobby.Players.Where(p => p.Team == ourTeam).Select(p => p.Battletag).ToArray();
        var stats = _playerData?.PlayerStats ?? _scan?.PlayerStats;
        if (stats != null)
            _playerData = new PlayerMawpData { PlayerStats = stats, AvailableBattletags = teammates };

        _lobbyStatus = $"lobby: {teammates.Length} teammates · {lobby.Map}";
        LobbyLog($"applied — you={me!.Battletag} ({me.ToonId}), team {ourTeam}: {string.Join(", ", teammates)}");
        UpdateYourBest();     // reflect the current account
        _ = RecomputeAsync(); // draw the auto-filled board + final win %, personalized for the team
    }

    // Populate the draft state from a completed lobby: our team → engine team 0
    // (so the board's mine/enemy sides are correct), enemy → team 1. The lobby
    // gives final rosters + bans by team, not pick order, so heroes are dropped
    // into that team's pick/ban slots in roster order — which is all the board
    // and the final-comp win probability need.
    private void FillDraftFromLobby(LobbyInfo lobby, int ourTeam)
    {
        _ourTeam = 0;
        if (HeroCatalog.MapIndex(lobby.Map) >= 0 && _map != lobby.Map)
        {
            _map = lobby.Map;
            _settingUp = true;
            MapCombo.SelectedItem = _map;
            _settingUp = false;
        }

        int[] StepsFor(int team, bool pick) =>
            Enumerable.Range(0, 16).Where(i => DraftOrder[i].Team == team && DraftOrder[i].IsPick == pick).ToArray();
        void Assign(int[] steps, IReadOnlyList<string> heroes)
        {
            for (int i = 0; i < steps.Length && i < heroes.Count; i++) _selections[steps[i]] = heroes[i];
        }

        _selections.Clear();
        var ourPicks = lobby.Players.Where(p => p.Team == ourTeam && p.Hero.Length > 0).Select(p => p.Hero).ToList();
        var enemyPicks = lobby.Players.Where(p => p.Team != ourTeam && p.Hero.Length > 0).Select(p => p.Hero).ToList();
        var ourBans = ourTeam == 0 ? lobby.BansBlue : lobby.BansRed;
        var enemyBans = ourTeam == 0 ? lobby.BansRed : lobby.BansBlue;

        Assign(StepsFor(0, true), ourPicks);
        Assign(StepsFor(1, true), enemyPicks);
        Assign(StepsFor(0, false), ourBans);
        Assign(StepsFor(1, false), enemyBans);
        _currentStep = 16; // draft complete
    }

    // Fill the board + recs from a live vision read of the draft screen. Left team
    // = ours (engine team 0), right = enemy. Recognized heroes are dropped into
    // their team's slots; the engine then recommends our next pick + win %.
    private async System.Threading.Tasks.Task ApplyVisionDraft(VisionDraft vd)
    {
        _ourTeam = 0; // HotS UI invariant: your team is ALWAYS the left column.
        if (vd.Map != null && HeroCatalog.MapIndex(vd.Map) >= 0 && _map != vd.Map)
        {
            _map = vd.Map; _settingUp = true; MapCombo.SelectedItem = _map; _settingUp = false;
        }

        // Heroes this read says are highlighted but NOT confirmed: the slot the team
        // on the clock is previewing, plus the big centre portrait. They look picked
        // but aren't, so they're barred from the board (and evict an earlier commit —
        // if the model now calls a hero pending, it was never locked).
        var pending = new HashSet<string>(vd.PendingLeft.Concat(vd.PendingRight), StringComparer.Ordinal);
        if (!string.IsNullOrEmpty(vd.PreviewHero)) pending.Add(vd.PreviewHero);

        // Counts the fixed 16-step draft order can't produce used to mean a bad read, back
        // when we sent the model the whole screen and it invented heroes from the centre
        // splash. Now that it only ever sees the two column crops, a verified-correct read
        // of a real screen came back L=5 R=3 — which this rule calls impossible. So it
        // logs and no longer drops: with clean input it was rejecting good data more often
        // than bad. RevokeReads still ages out anything genuinely wrong.
        int LockedCount(IReadOnlyList<string> team) =>
            team.Count(h => HeroCatalog.HeroIndex(h) >= 0 && !pending.Contains(h));
        int left = LockedCount(vd.LeftTeam), right = LockedCount(vd.RightTeam);
        if (!ReachablePickCounts.Contains((left, right)))
            DraftLog($"vision read has off-order pick counts L={left} R={right} (accepted)");

        // Collapse this read's LOCKED heroes into a hero -> slot map (valid catalog
        // heroes only; first side wins so a hero never lands on both sides).
        var seen = new Dictionary<string, (int side, bool pick)>(StringComparer.Ordinal);
        void Note(IReadOnlyList<string> heroes, int side, bool pick)
        {
            foreach (var h in heroes)
                if (HeroCatalog.HeroIndex(h) >= 0 && !pending.Contains(h) && !seen.ContainsKey(h))
                    seen[h] = (side, pick);
        }
        Note(vd.LeftTeam, 0, true);
        Note(vd.RightTeam, 1, true);
        Note(vd.BansLeft, 0, false);
        Note(vd.BansRight, 1, false);

        List<string> ListFor((int side, bool pick) slot) => slot switch
        {
            (0, true) => _vOurPicks,
            (1, true) => _vEnemyPicks,
            (0, false) => _vOurBans,
            _ => _vEnemyBans,
        };
        void Uncommit(string h)
        {
            _vCandidates.Remove(h);
            _vMisses.Remove(h);
            if (_vCommitted.Remove(h, out var slot)) ListFor(slot).Remove(h);
        }

        foreach (var h in pending) Uncommit(h);

        // Count consecutive reads in the same slot; promote at ConfirmReads.
        foreach (var kv in seen)
        {
            var h = kv.Key; var slot = kv.Value;
            _vMisses.Remove(h);
            if (_vCommitted.ContainsKey(h)) continue;
            int hits = _vCandidates.TryGetValue(h, out var c) && c.slot == slot ? c.hits + 1 : 1;
            if (hits >= ConfirmReads)
            {
                _vCandidates.Remove(h);
                _vCommitted[h] = slot;
                var list = ListFor(slot);
                if (!list.Contains(h)) list.Add(h);
            }
            else _vCandidates[h] = (slot, hits);
        }

        // A candidate that didn't survive this read starts over; a committed hero
        // that stays missing for RevokeReads was a preview (or a bad read) and goes.
        foreach (var h in _vCandidates.Keys.ToList())
            if (!seen.ContainsKey(h)) _vCandidates.Remove(h);
        foreach (var h in _vCommitted.Keys.ToList())
        {
            if (seen.ContainsKey(h)) continue;
            int miss = _vMisses.TryGetValue(h, out var m) ? m + 1 : 1;
            if (miss >= RevokeReads) Uncommit(h);
            else _vMisses[h] = miss;
        }

        int[] StepsFor(int team, bool pick) =>
            Enumerable.Range(0, 16).Where(i => DraftOrder[i].Team == team && DraftOrder[i].IsPick == pick).ToArray();
        void Assign(int[] steps, IReadOnlyList<string> heroes)
        {
            for (int i = 0; i < steps.Length && i < heroes.Count; i++) _selections[steps[i]] = heroes[i];
        }

        // Rebuild the board from the confirmed rosters.
        _selections.Clear();
        Assign(StepsFor(0, true), _vOurPicks);
        Assign(StepsFor(1, true), _vEnemyPicks);
        Assign(StepsFor(0, false), _vOurBans);
        Assign(StepsFor(1, false), _vEnemyBans);
        // Current decision = first pick step not yet filled. Picks sit in their
        // draft-order pick slots, so the pick counts before this step match
        // exactly — the engine state stays consistent (no index crash), and
        // RecomputeAsync yields live next-pick recs + win %.
        _currentStep = 16;
        for (int i = 0; i < 16; i++)
            if (DraftOrder[i].IsPick && !_selections.ContainsKey(i)) { _currentStep = i; break; }

        // Heroes a player has SHOWN their team but not locked in. These are display-only:
        // they go on the board so you can see what's coming, but never into _selections,
        // so _currentStep and the engine recs are exactly what they'd be without them.
        // No confirmation counting either — a hover is transient and worth showing at once;
        // if it's withdrawn it simply stops appearing on the next read.
        _vShowing.Clear();
        void Show(IReadOnlyList<string> heroes, int team)
        {
            var hero = heroes.FirstOrDefault(h => HeroCatalog.HeroIndex(h) >= 0 && !_vCommitted.ContainsKey(h));
            if (hero == null) return;
            foreach (int step in StepsFor(team, true))
                if (!_selections.ContainsKey(step)) { _vShowing[step] = hero; return; }
        }
        Show(vd.PendingLeft, 0);
        Show(vd.PendingRight, 1);

        _lobbyStatus = $"live draft (vision) · {_map}";
        await RecomputeAsync();
    }

    // Clear an auto-filled draft back to an empty, un-personalized state (called
    // when a match ends, so the next game doesn't start on stale data).
    private void ResetForNewGame()
    {
        _autoFilled = false;
        _selections.Clear();
        _vOurPicks.Clear(); _vEnemyPicks.Clear(); _vOurBans.Clear(); _vEnemyBans.Clear();
        _vCommitted.Clear(); _vCandidates.Clear(); _vMisses.Clear(); _vShowing.Clear();
        _currentStep = 0;
        _lobbyStatus = "";
        if (_scan?.LocalBattletag is string bt && _scan.PlayerStats.Count > 0)
            _playerData = new PlayerMawpData { PlayerStats = _scan.PlayerStats, AvailableBattletags = new[] { bt } };
        if (SearchBox != null) SearchBox.Text = "";
        PopulateHeroGrid();
        _ = RecomputeAsync();
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
            Threats.Clear();
            ThreatSection.Visibility = Visibility.Collapsed;
            // Completed draft (e.g. auto-filled from the lobby): still show the final win %.
            if (_sessions != null && _data != null && _currentStep >= 16)
            {
                var full = BuildInput();
                if (full.Team0Picks.Count == 5 && full.Team1Picks.Count == 5)
                {
                    var wpSessions = _sessions; var wpData = _data; var wpMap = _map; var wpTier = _tier;
                    var t0 = full.Team0Picks; var t1 = full.Team1Picks;
                    float p0 = await System.Threading.Tasks.Task.Run(() =>
                        HotsFever.DraftEngine.Models.WinProbability.Get(wpSessions, t0, t1, wpMap, wpTier, wpData));
                    double ours = _ourTeam == 0 ? p0 : 1 - p0;
                    WinProbText.Text = (int)Math.Round(ours * 100) + "%";
                }
            }
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

        // On the ENEMY's turn, also price what they're likely to take. The pick they
        // make is the thing you're about to have to play against, and on a ban step
        // it's the list you'd want to pre-empt.
        bool enemyTurn = DraftOrder[_currentStep].Team != _ourTeam;
        await UpdateThreatsAsync(enemyTurn ? input : null, isBan, sessions, data);
    }

    private async System.Threading.Tasks.Task UpdateThreatsAsync(
        MctsSearch.Input? input, bool isBan, OnnxSessions sessions, DraftData? data)
    {
        Threats.Clear();
        if (input == null)
        {
            ThreatSection.Visibility = Visibility.Collapsed;
            return;
        }

        IReadOnlyList<AiInference.OpponentPrediction> preds;
        try
        {
            var inp = input;
            preds = await System.Threading.Tasks.Task.Run(() =>
                AiInference.GetOpponentPredictions(sessions, inp, isBan, data, topK: 4));
        }
        catch (Exception ex)
        {
            DraftLog("threats failed: " + ex.Message);
            ThreatSection.Visibility = Visibility.Collapsed;
            return;
        }

        var red = new SolidColorBrush(Color.FromArgb(0xFF, 0xFF, 0x6B, 0x6B));
        var green = new SolidColorBrush(Color.FromArgb(0xFF, 0x4F, 0xFF, 0xB0));
        var grey = new SolidColorBrush(Color.FromArgb(0xFF, 0x8B, 0x9B, 0xC8));

        foreach (var p in preds)
        {
            string impact = p.ImpactPp is double pp
                ? (pp >= 0 ? "+" : "") + pp.ToString("0.0") + "pp"
                : "—";
            Threats.Add(new ThreatItem
            {
                Portrait = Short(p.Hero),
                Hero = p.Hero,
                Subtitle = $"{Math.Round(p.Probability * 100)}% likely  ·  {RoleName(p.Hero)}",
                Impact = impact,
                // Colour by what it does to US: their gain is our loss.
                ImpactBrush = p.ImpactPp is double v ? (v <= -1 ? red : v >= 1 ? green : grey) : grey,
            });
        }

        ThreatHeader.Text = isBan ? "ENEMY LIKELY BANS" : "ENEMY THREATS";
        ThreatSection.Visibility = Threats.Count > 0 ? Visibility.Visible : Visibility.Collapsed;
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
            // Cho and Gall go together: taking either makes both unavailable.
            TakenHeroes = HeroCatalog.ExpandChoGall(t0.Concat(t1).Concat(bans)),
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
            bool showing = false;
            if (hero == null && _vShowing.TryGetValue(s, out var shown)) { hero = shown; showing = true; }
            bool mine = team == _ourTeam;
            var slot = MakeSlot(hero, s == _currentStep, mine, showing);
            if (!isPick) (mine ? BansMine : BansEnemy).Add(slot);
            else (mine ? PicksMine : PicksEnemy).Add(slot);
        }
    }

    // showing = the player has revealed this hero to their team but hasn't locked it in.
    // Drawn in amber with a faded fill so it never reads as a settled pick.
    private static SlotVM MakeSlot(string? hero, bool isCurrent, bool mine, bool showing = false)
    {
        bool filled = hero != null;
        var mineFill = Color.FromArgb(0x40, 0x4A, 0x9E, 0xFF);   // blue
        var enemyFill = Color.FromArgb(0x40, 0xFF, 0x6B, 0x6B);  // red
        var emptyFill = Color.FromArgb(0x12, 0xFF, 0xFF, 0xFF);
        var accent = Color.FromArgb(0xFF, 0x4F, 0xFF, 0xB0);     // current = green
        var amber = Color.FromArgb(0xFF, 0xFF, 0xC1, 0x4F);      // showing = amber
        var showFill = mine
            ? Color.FromArgb(0x1E, 0x4A, 0x9E, 0xFF)
            : Color.FromArgb(0x1E, 0xFF, 0x6B, 0x6B);

        var border = showing ? amber : (isCurrent ? accent : Color.FromArgb(0, 0, 0, 0));

        return new SlotVM
        {
            Portrait = filled ? Short(hero!) : "",
            Name = showing ? hero + " ?" : (filled ? hero! : (isCurrent ? "…" : "")),
            Background = new SolidColorBrush(showing ? showFill : (filled ? (mine ? mineFill : enemyFill) : emptyFill)),
            BorderColor = new SolidColorBrush(border),
            BorderT = new Thickness(showing || isCurrent ? 2 : 0),
            NameForeground = new SolidColorBrush(showing
                ? amber
                : (filled ? Color.FromArgb(0xFF, 0xEA, 0xF1, 0xFB)
                          : Color.FromArgb(0xFF, 0x7F, 0x93, 0xB0))),
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
        ReassertTopMost(); // Resize resets z-order via the presenter — restore it immediately.
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

    // Prefer assets bundled next to the app (shipped build); fall back to the
    // repo layout when running from a dev checkout.
    private static string ResolveAssetDir(string bundledSubDir, string sentinelFile, string repoRelativeDir)
    {
        var bundled = System.IO.Path.Combine(AppContext.BaseDirectory, "Assets", bundledSubDir);
        if (System.IO.File.Exists(System.IO.Path.Combine(bundled, sentinelFile))) return bundled;
        return LocateDir(repoRelativeDir, sentinelFile);
    }

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
        ReassertTopMost();
    }

    // ── Window styling ───────────────────────────────────────────────

    // WinUI's OverlappedPresenter re-applies its window state on layout/resize
    // passes and clears the raw WS_EX_TOPMOST bit, so we re-assert BOTH the
    // presenter-native flag (which WinUI actually enforces) and the raw call.
    private void ReassertTopMost()
    {
        // Check the REAL window bit, not the presenter's cached IsAlwaysOnTop —
        // after a resize the cache can read true while the window is actually
        // non-topmost (WinUI keeps re-applying the stale state, and a bare
        // SetWindowPos gets overwritten each frame). When the bit is genuinely
        // missing, force WinUI to re-apply by toggling the presenter off→on.
        bool isTop = (NativeMethods.GetExStyle(_hWnd) & NativeMethods.WS_EX_TOPMOST) != 0;
        if (!isTop && _presenter != null)
        {
            _presenter.IsAlwaysOnTop = false;
            _presenter.IsAlwaysOnTop = true;
        }
        NativeMethods.SetTopMost(_hWnd);
    }

    private void ConfigureAsOverlay()
    {
        var hWnd = WindowNative.GetWindowHandle(this);
        _hWnd = hWnd;
        var windowId = Win32Interop.GetWindowIdFromWindow(hWnd);
        var appWindow = AppWindow.GetFromWindowId(windowId);
        _appWindow = appWindow;

        if (appWindow.Presenter is OverlappedPresenter presenter)
        {
            _presenter = presenter;
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
