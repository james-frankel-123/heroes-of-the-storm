using HotsFever.DraftEngine.Data;
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

        var mcts = MctsSearch.Run(sessions, input, rng, options, evaluateTerminal);

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
}
