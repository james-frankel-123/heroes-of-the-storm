using HotsFever.DraftEngine.Data;
using HotsFever.DraftEngine.Encoding;

namespace HotsFever.DraftEngine.Models;

/// <summary>
/// Partial-draft win probability, symmetrized — a port of getPartialProjection in
/// ai-inference.ts. Same 283 enriched features as <see cref="WinProbability"/> plus a
/// step embedding, and the same normal/team-swapped averaging so that
/// P(t0,t1) + P(t1,t0) = 1 exactly.
///
/// Why this exists rather than reusing the terminal judge mid-draft: the terminal
/// model only ever saw finished comps, so on a half-built draft it barely moves. The
/// web engine measured the policy value head as effectively a state-insensitive
/// constant and replaced it with this model, which stays responsive at every stage and
/// converges to the terminal evaluator by the last pick. The overlay was still showing
/// the older number, so the same draft could read differently in the two products.
/// </summary>
public static class PartialWinProbability
{
    /// <summary>
    /// P(team0 wins) from a partial draft state at <paramref name="step"/> (0-15).
    /// Falls back to the terminal judge when partial_wp.onnx isn't loaded.
    /// </summary>
    public static float Get(OnnxSessions sessions, IReadOnlyList<string> team0, IReadOnlyList<string> team1,
        string map, string tier, int step, DraftData? draftData)
    {
        if (!sessions.HasPartialWinProbability)
            return WinProbability.Get(sessions, team0, team1, map, tier, draftData);

        float RunPartial(IReadOnlyList<string> t0h, IReadOnlyList<string> t1h)
        {
            var input = StateEncoder.EncodeWpBase(t0h, t1h, map, tier); // 283 dims, enriched zero
            if (draftData != null)
            {
                var enriched = EnrichedFeatures.Compute(t0h, t1h, map, draftData);
                Array.Copy(enriched, 0, input, StateEncoder.WpBaseDims, EnrichedFeatures.Dims);
            }
            return sessions.RunPartialWinProbability(input, step);
        }

        float pNormal = RunPartial(team0, team1);
        float pSwapped = RunPartial(team1, team0);
        return (pNormal + (1 - pSwapped)) / 2f;
    }
}
