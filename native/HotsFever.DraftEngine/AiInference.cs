using HotsFever.DraftEngine.Data;
using HotsFever.DraftEngine.Encoding;
using HotsFever.DraftEngine.Models;
using HotsFever.DraftEngine.Rng;
using HotsFever.DraftEngine.Scoring;
using HotsFever.DraftEngine.Search;

namespace HotsFever.DraftEngine;

/// <summary>A single recommended hero — mirrors AIRecommendation in ai-inference.ts.</summary>
public sealed record AiRecommendation(
    string Hero,
    /// <summary>Policy/visit prior (root-visit fraction).</summary>
    double Prior,
    /// <summary>P(our team wins) if we take this hero, incl. MAWP adjustments, clamped [0,1].</summary>
    double WinProb,
    double MawpAdj,
    string? SuggestedPlayer);

/// <summary>
/// Top-level recommendation orchestration — a port of getAIRecommendations in
/// src/lib/draft/ai-inference.ts. Runs the neural MCTS with the WP model as the
/// terminal-draft evaluator (when DraftData is available), then layers in MAWP
/// player adjustments and ranks the results.
/// </summary>
public static class AiInference
{
    public sealed record Result(IReadOnlyList<AiRecommendation> Recommendations, double ValueEstimate, int Sims);

    public static Result GetRecommendations(
        OnnxSessions sessions,
        MctsSearch.Input input,
        bool isBanStep,
        ISeedableRng rng,
        MctsSearch.Options? options = null,
        DraftData? draftData = null,
        PlayerMawpData? playerData = null,
        int topK = 15)
    {
        options ??= new MctsSearch.Options();

        double teamMawpAdj = Mawp.ComputeTeamMawpAdjustment(
            input.Team0Picks.Concat(input.Team1Picks), playerData);

        // Terminal (complete-draft) leaves use the dedicated WP model when we
        // have the stat tables; otherwise MCTS falls back to the policy value head.
        MctsSearch.TerminalEvaluator? evaluateTerminal = draftData != null
            ? (t0, t1) => WinProbability.Get(sessions, t0, t1, input.Map, input.Tier, draftData)
            : null;

        // Non-terminal leaves use the partial-draft judge, so every leaf in the search
        // is scored on ONE scale that converges to the terminal judge by the last pick.
        // Without it the policy value head answers here, and it barely responds to the
        // board — which made backed-up Q, and therefore the displayed deltas, mushy.
        MctsSearch.PartialEvaluator? evaluatePartial = draftData != null && sessions.HasPartialWinProbability
            ? (t0, t1, lastActionIdx) =>
                PartialWinProbability.Get(sessions, t0, t1, input.Map, input.Tier, lastActionIdx, draftData)
            : null;

        var mcts = MctsSearch.Run(sessions, input, rng, options, evaluateTerminal, evaluatePartial);

        var recs = mcts.Recommendations
            .Select(r =>
            {
                var (adj, player) = isBanStep
                    ? (0.0, (string?)null)
                    : Unpack(Mawp.ComputePlayerAdjustment(r.Hero, playerData));
                double winProb = Math.Clamp(r.Q + adj + teamMawpAdj, 0, 1);
                return new AiRecommendation(r.Hero, r.Visits, winProb, adj, player);
            })
            // recommendationSorter: prior desc, then winProb desc, then hero (ordinal —
            // exact sort parity is not required per the M1 plan).
            .OrderByDescending(r => r.Prior)
            .ThenByDescending(r => r.WinProb)
            .ThenBy(r => r.Hero, StringComparer.Ordinal)
            .Take(topK)
            .ToList();

        double valueEstimate = Math.Clamp(mcts.ValueEstimate + teamMawpAdj, 0, 1);
        return new Result(recs, valueEstimate, mcts.Sims);
    }

    private static (double, string?) Unpack(Mawp.Adjustment a) => (a.Value, a.Player);

    /// <summary>
    /// What the ENEMY is likely to take next, and what it would cost us.
    /// <paramref name="Probability"/> is the opponent model's likelihood (0-1);
    /// <paramref name="ImpactPp"/> is the change to OUR win chance in percentage points
    /// if they take it, or null on a ban step (a ban removes a hero rather than adding
    /// one to their comp, so the same before/after comparison doesn't apply).
    /// </summary>
    public sealed record OpponentPrediction(string Hero, double Probability, double? ImpactPp);

    /// <summary>
    /// Port of getGenericDraftPredictions: run the opponent model over the current state
    /// and rank the heroes it expects the enemy to take, then price each one.
    ///
    /// The web prices impact with its statistical team estimate (computeTeamWinEstimate).
    /// We use the engine's own partial-draft judge instead, because that is what the
    /// overlay's win % is already showing — mixing a second, differently-scaled estimator
    /// into the same panel would make the two numbers disagree on screen.
    /// </summary>
    public static IReadOnlyList<OpponentPrediction> GetOpponentPredictions(
        OnnxSessions sessions,
        MctsSearch.Input input,
        bool isBanStep,
        DraftData? draftData = null,
        int topK = 6)
    {
        var state = new AiDraftState
        {
            Team0Picks = input.Team0Picks,
            Team1Picks = input.Team1Picks,
            Bans = input.Bans,
            Map = input.Map,
            Tier = input.Tier,
            Step = Math.Clamp(input.Step, 0, 15),
            IsPick = !isBanStep,
            OurTeam = input.OurTeam,
        };

        var gdState = StateEncoder.EncodeStateForGd(state);
        var mask = StateEncoder.BuildValidMask(input.TakenHeroes);
        var logits = sessions.RunGenericDraft(gdState, mask);
        var probs = SoftmaxMasked(logits, mask);

        var ranked = new List<(string Hero, double P)>();
        for (int i = 0; i < HeroCatalog.NumHeroes; i++)
            if (mask[i] > 0 && probs[i] > 0.001f)
                ranked.Add((HeroCatalog.Heroes[i], probs[i]));
        ranked.Sort((a, b) => b.P.CompareTo(a.P));

        int enemyTeam = 1 - input.OurTeam;
        var ourPicks = input.OurTeam == 0 ? input.Team0Picks : input.Team1Picks;
        var enemyPicks = enemyTeam == 0 ? input.Team0Picks : input.Team1Picks;

        double? before = null;
        if (!isBanStep && draftData != null)
            before = OurWinPct(sessions, ourPicks, enemyPicks, input, draftData, input.Step);

        // Price a wider candidate set than we display: the model's MOST LIKELY pick is
        // frequently not its most DAMAGING one (measured on a mid-draft state: Brightwing
        // ranked first at 8.3% for +1.1pp to us, while Falstad sat sixth at 2.9% for
        // -8.3pp). Ranking the shown rows by damage answers the question a threat list is
        // actually asked — what should I be worried about — rather than just restating the
        // model's ordering.
        var candidates = new List<OpponentPrediction>();
        foreach (var (hero, p) in ranked.Take(Math.Max(topK * 2, topK)))
        {
            double? impact = null;
            if (before is double b)
            {
                var withHero = enemyPicks.Concat(new[] { hero }).ToList();
                double after = OurWinPct(sessions, ourPicks, withHero, input, draftData!, input.Step + 1);
                impact = Math.Round((after - b) * 100.0, 1);
            }
            candidates.Add(new OpponentPrediction(hero, p, impact));
        }

        // Ban steps carry no impact (a ban removes a hero rather than adding one to their
        // comp), so there they stay in likelihood order.
        return candidates.Any(c => c.ImpactPp.HasValue)
            ? candidates.OrderBy(c => c.ImpactPp ?? 0).Take(topK).ToList()
            : candidates.Take(topK).ToList();
    }

    /// <summary>P(we win) from the partial judge, oriented to our team.</summary>
    private static double OurWinPct(OnnxSessions sessions, IReadOnlyList<string> ourPicks,
        IReadOnlyList<string> enemyPicks, MctsSearch.Input input, DraftData draftData, int step)
    {
        var t0 = input.OurTeam == 0 ? ourPicks : enemyPicks;
        var t1 = input.OurTeam == 0 ? enemyPicks : ourPicks;
        float p0 = PartialWinProbability.Get(sessions, t0, t1, input.Map, input.Tier,
            Math.Clamp(step, 0, 15), draftData);
        return input.OurTeam == 0 ? p0 : 1 - p0;
    }

    /// <summary>Masked softmax — same shape as the TS softmaxMasked.</summary>
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
