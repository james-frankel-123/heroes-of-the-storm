using System.Diagnostics;
using HotsFever.DraftEngine.Encoding;
using HotsFever.DraftEngine.Models;
using HotsFever.DraftEngine.Rng;

namespace HotsFever.DraftEngine.Search;

/// <summary>
/// AlphaZero-style MCTS — a faithful port of src/lib/draft/mcts-search.ts.
///   * policy net gives tree priors (tempered) + value estimates
///   * GD model samples opponent moves during descent/rollout (via ISeedableRng)
///   * optional WP evaluator scores terminal (complete) drafts
///
/// Determinism: with a <see cref="SequenceRng"/> and an uncapped time budget
/// (fixed sim count), a run is exactly reproducible and comparable to the web
/// engine's mock-RNG mode. In production, pass <see cref="SystemRng"/> and the
/// real time budget.
/// </summary>
public static class MctsSearch
{
    private const int PolicyDim = StateEncoder.PolicyDims; // 290
    private const int NumHeroes = HeroCatalog.NumHeroes;   // 90
    private const double CPuct = 2.0;
    private const double PriorTemp = 3.0;

    // The fixed 16-step draft order: (team, isPick). Mirrors DRAFT_ORDER in the TS.
    private static readonly (int Team, bool IsPick)[] DraftOrder =
    {
        (0, false), (1, false), (0, false), (1, false),
        (0, true),  (1, true),  (1, true),  (0, true),  (0, true),
        (1, false), (0, false),
        (1, true),  (1, true),  (0, true),  (0, true),  (1, true),
    };

    public sealed class Options
    {
        public int MinSims { get; init; } = 50;
        public int MaxSims { get; init; } = 200;
        /// <summary>Wall-clock budget in ms; use double.PositiveInfinity to run exactly MaxSims (test mode).</summary>
        public double TimeBudgetMs { get; init; } = 1500;
    }

    public sealed class Input
    {
        public IReadOnlyList<string> Team0Picks { get; init; } = Array.Empty<string>();
        public IReadOnlyList<string> Team1Picks { get; init; } = Array.Empty<string>();
        public IReadOnlyList<string> Bans { get; init; } = Array.Empty<string>();
        public IReadOnlyList<string> TakenHeroes { get; init; } = Array.Empty<string>();
        public string Map { get; init; } = "";
        public string Tier { get; init; } = "mid";
        public int Step { get; init; }
        public int OurTeam { get; init; }
    }

    public sealed record Recommendation(string Hero, double Visits, double Q);

    public sealed record Result(IReadOnlyList<Recommendation> Recommendations, double ValueEstimate, int Sims);

    /// <summary>Optional terminal evaluator: (team0Heroes, team1Heroes) → P(team0 wins).</summary>
    public delegate float TerminalEvaluator(IReadOnlyList<string> team0, IReadOnlyList<string> team1);

    /// <summary>
    /// Optional partial-draft leaf evaluator for INCOMPLETE states:
    /// (team0Heroes, team1Heroes, lastActionIdx) → P(team0 wins). When supplied it
    /// replaces the policy value head at non-terminal leaves — the head measured as
    /// state-insensitive, so backed-up Q barely moved with the board. Ban values then
    /// emerge from search dynamics, since a banned hero is unavailable in every branch.
    /// </summary>
    public delegate float PartialEvaluator(IReadOnlyList<string> team0, IReadOnlyList<string> team1, int lastActionIdx);

    private sealed class State
    {
        public List<int> Team0Picks = new();
        public List<int> Team1Picks = new();
        public List<int> Bans = new();
        public HashSet<int> Taken = new();
        public int Step;
        public string Map = "";
        public string Tier = "mid";
        public int OurTeam;

        public State Clone() => new()
        {
            Team0Picks = new List<int>(Team0Picks),
            Team1Picks = new List<int>(Team1Picks),
            Bans = new List<int>(Bans),
            Taken = new HashSet<int>(Taken),
            Step = Step, Map = Map, Tier = Tier, OurTeam = OurTeam,
        };

        public void Apply(int heroIdx)
        {
            var (team, isPick) = DraftOrder[Step];
            Taken.Add(heroIdx);
            if (!isPick) Bans.Add(heroIdx);
            else if (team == 0) Team0Picks.Add(heroIdx);
            else Team1Picks.Add(heroIdx);
            Step++;
        }
    }

    private sealed class Node
    {
        public int Action;
        public Node? Parent;
        // Children kept in insertion order (ascending action index) to match the
        // JS Map iteration order that MCTS tie-breaks depend on.
        public readonly List<Node> Children = new();
        public int VisitCount;
        public double ValueSum;
        public double Prior;
        public bool IsExpanded;

        public Node(int action, Node? parent, double prior)
        {
            Action = action; Parent = parent; Prior = prior;
        }

        public double Ucb()
        {
            if (Parent == null) return 0;
            double q = VisitCount == 0 ? 0 : ValueSum / VisitCount;
            return q + CPuct * Prior * Math.Sqrt(Parent.VisitCount) / (1 + VisitCount);
        }
    }

    public static Result Run(OnnxSessions sessions, Input input, ISeedableRng rng, Options? options = null,
        TerminalEvaluator? evaluateTerminal = null, PartialEvaluator? evaluatePartial = null)
    {
        options ??= new Options();
        int ourTeam = input.OurTeam;

        // Partial-draft leaf value, already oriented to OUR team.
        Func<State, double>? evalPartialOur = evaluatePartial == null ? null : s =>
        {
            float p = evaluatePartial(Names(s.Team0Picks), Names(s.Team1Picks), Math.Clamp(s.Step - 1, 0, 15));
            return ourTeam == 0 ? p : 1 - p;
        };

        var root = new State
        {
            Team0Picks = ToIdx(input.Team0Picks),
            Team1Picks = ToIdx(input.Team1Picks),
            Bans = ToIdx(input.Bans),
            Taken = new HashSet<int>(ToIdx(input.TakenHeroes)),
            Step = input.Step, Map = input.Map, Tier = input.Tier, OurTeam = ourTeam,
        };

        var rootNode = new Node(-1, null, 0);
        var (rootState, rootMask) = Encode(root);
        var (priors, rootValue) = RunPolicy(sessions, rootState, rootMask);

        rootNode.IsExpanded = true;
        for (int a = 0; a < NumHeroes; a++)
            if (rootMask[a] > 0) AddChild(rootNode, new Node(a, rootNode, priors[a]));

        var sw = Stopwatch.StartNew();
        bool timeCapped = !double.IsPositiveInfinity(options.TimeBudgetMs);
        int simsRun = 0;

        for (int sim = 0; sim < options.MaxSims; sim++)
        {
            if (timeCapped && sim >= options.MinSims && sw.Elapsed.TotalMilliseconds > options.TimeBudgetMs) break;
            simsRun++;

            var node = rootNode;
            var scratch = root.Clone();

            // Selection
            while (node.IsExpanded && scratch.Step < 16)
            {
                int currentTeam = DraftOrder[scratch.Step].Team;
                if (currentTeam == ourTeam)
                {
                    double bestScore = double.NegativeInfinity;
                    Node? bestChild = null;
                    foreach (var child in node.Children)
                    {
                        double score = child.Ucb();
                        if (score > bestScore) { bestScore = score; bestChild = child; }
                    }
                    if (bestChild == null) break;
                    scratch.Apply(bestChild.Action);
                    node = bestChild;
                }
                else
                {
                    scratch.Apply(RunGd(sessions, scratch, rng));
                }
            }

            // Roll forward through opponent steps so the leaf sits at OUR decision (or terminal).
            while (scratch.Step < 16 && DraftOrder[scratch.Step].Team != ourTeam)
                scratch.Apply(RunGd(sessions, scratch, rng));

            double value;
            if (scratch.Step >= 16)
            {
                if (evaluateTerminal != null)
                {
                    float wpTeam0 = evaluateTerminal(Names(scratch.Team0Picks), Names(scratch.Team1Picks));
                    value = ourTeam == 0 ? wpTeam0 : 1 - wpTeam0;
                }
                else
                {
                    var (s, m) = Encode(scratch);
                    value = RunPolicy(sessions, s, m).Value;
                }
            }
            else if (!node.IsExpanded)
            {
                // Expand at our decision point: policy priors for exploration, the
                // partial-draft evaluator (when provided) for the backed-up value.
                var (s, m) = Encode(scratch);
                var (leafPriors, v) = RunPolicy(sessions, s, m);
                value = evalPartialOur != null ? evalPartialOur(scratch) : v;
                node.IsExpanded = true;
                for (int a = 0; a < NumHeroes; a++)
                    if (m[a] > 0) AddChild(node, new Node(a, node, leafPriors[a]));
            }
            else if (evalPartialOur != null)
            {
                value = evalPartialOur(scratch);
            }
            else
            {
                var (s, m) = Encode(scratch);
                value = RunPolicy(sessions, s, m).Value;
            }

            // Backpropagate
            Node? current = node;
            while (current != null)
            {
                current.VisitCount++;
                current.ValueSum += value;
                current = current.Parent;
            }
        }

        // Build recommendations: visit fraction + mean action value, sorted by visits desc.
        int visitSum = rootNode.Children.Sum(c => c.VisitCount);
        var recs = rootNode.Children
            .Where(c => c.VisitCount > 0)
            .Select(c => new Recommendation(
                HeroCatalog.Heroes[c.Action],
                visitSum > 0 ? (double)c.VisitCount / visitSum : 0,
                c.ValueSum / c.VisitCount))
            .OrderByDescending(r => r.Visits) // stable → ties keep ascending-action order
            .ToList();

        double valueEstimate = rootNode.VisitCount > 0 ? rootNode.ValueSum / rootNode.VisitCount : rootValue;
        return new Result(recs, valueEstimate, simsRun);
    }

    private static void AddChild(Node parent, Node child) => parent.Children.Add(child);

    private static List<int> ToIdx(IReadOnlyList<string> names)
    {
        var list = new List<int>(names.Count);
        foreach (var n in names)
        {
            int i = HeroCatalog.HeroIndex(n);
            if (i >= 0) list.Add(i);
        }
        return list;
    }

    private static List<string> Names(List<int> idx)
    {
        var list = new List<string>(idx.Count);
        foreach (var i in idx) list.Add(HeroCatalog.Heroes[i]);
        return list;
    }

    /// <summary>Build the 290-dim policy state + 90-dim mask directly from index-space state.</summary>
    private static (float[] State, float[] Mask) Encode(State s)
    {
        var v = new float[PolicyDim];
        foreach (var i in s.Team0Picks) v[i] = 1f;
        foreach (var i in s.Team1Picks) v[90 + i] = 1f;
        foreach (var i in s.Bans) v[180 + i] = 1f;

        int mapIdx = HeroCatalog.MapIndex(s.Map);
        if (mapIdx >= 0) v[270 + mapIdx] = 1f;
        int tierIdx = HeroCatalog.TierIndex(s.Tier);
        if (tierIdx >= 0) v[284 + tierIdx] = 1f;

        bool isPick = s.Step < 16 ? DraftOrder[s.Step].IsPick : true;
        v[287] = s.Step / 15.0f;
        v[288] = isPick ? 1.0f : 0.0f;
        v[289] = s.OurTeam;

        var mask = new float[NumHeroes];
        Array.Fill(mask, 1f);
        foreach (var i in s.Taken) mask[i] = 0f;
        return (v, mask);
    }

    private static (float[] Priors, float Value) RunPolicy(OnnxSessions sessions, float[] state, float[] mask)
    {
        var result = sessions.RunPolicy(state, mask);
        // Temper the (near-deterministic) policy logits before softmax so MCTS explores.
        var tempered = new float[result.PolicyLogits.Length];
        for (int i = 0; i < tempered.Length; i++) tempered[i] = (float)(result.PolicyLogits[i] / PriorTemp);
        return (SoftmaxMasked(tempered, mask), result.Value);
    }

    private static int RunGd(OnnxSessions sessions, State s, ISeedableRng rng)
    {
        var (state, mask) = Encode(s);
        // GD input is the policy state minus the trailing ourTeam bit (289 dims).
        var gdState = new float[StateEncoder.GdDims];
        Array.Copy(state, gdState, StateEncoder.GdDims);

        var logits = sessions.RunGenericDraft(gdState, mask);
        var probs = SoftmaxMasked(logits, mask);

        double r = rng.NextDouble();
        double cumSum = 0;
        for (int i = 0; i < NumHeroes; i++)
        {
            cumSum += probs[i];
            if (r < cumSum) return i;
        }
        for (int i = 0; i < NumHeroes; i++) if (mask[i] > 0) return i;
        return 0;
    }

    /// <summary>Masked softmax — bit-for-bit the same shape as the TS softmaxMasked.</summary>
    private static float[] SoftmaxMasked(float[] logits, float[] mask)
    {
        var result = new float[logits.Length];
        float maxVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (mask[i] > 0 && logits[i] > maxVal) maxVal = logits[i];

        float sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (mask[i] > 0)
            {
                result[i] = (float)Math.Exp(logits[i] - maxVal);
                sum += result[i];
            }
        }
        if (sum > 0)
            for (int i = 0; i < result.Length; i++) result[i] /= sum;
        return result;
    }
}
