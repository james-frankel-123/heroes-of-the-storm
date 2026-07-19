using HotsFever.DraftEngine.Data;
using HotsFever.DraftEngine.Encoding;

namespace HotsFever.DraftEngine.Models;

/// <summary>
/// Win-Probability model, symmetrized — a port of getWinProbability in
/// ai-inference.ts. Runs the model twice (normal + team-swapped) and averages
/// so that P(t0,t1) + P(t1,t0) = 1 exactly, removing the model's team-order bias.
/// When <paramref name="draftData"/> is supplied, the 86 enriched features are
/// filled in; otherwise they stay zero (matching the web engine's fallback).
/// </summary>
public static class WinProbability
{
    /// <summary>P(team0 wins).</summary>
    public static float Get(OnnxSessions sessions, IReadOnlyList<string> team0, IReadOnlyList<string> team1,
        string map, string tier, DraftData? draftData)
    {
        float RunWp(IReadOnlyList<string> t0h, IReadOnlyList<string> t1h)
        {
            var input = StateEncoder.EncodeWpBase(t0h, t1h, map, tier); // 283 dims, enriched zero
            if (draftData != null)
            {
                var enriched = EnrichedFeatures.Compute(t0h, t1h, map, draftData);
                Array.Copy(enriched, 0, input, StateEncoder.WpBaseDims, EnrichedFeatures.Dims);
            }
            return sessions.RunWinProbability(input);
        }

        float pNormal = RunWp(team0, team1);
        float pSwapped = RunWp(team1, team0);
        return (pNormal + (1 - pSwapped)) / 2f;
    }
}
