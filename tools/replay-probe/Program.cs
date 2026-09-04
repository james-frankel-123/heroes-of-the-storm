using System;
using System.Linq;
using Heroes.ReplayParser;

// Probe: what granular data does a real .StormReplay actually yield?
class Program
{
    static void Main(string[] args)
    {
        var path = args[0];
        var (result, replay) = DataParser.ParseReplay(path, false, ParseOptions.FullParsing);
        Console.WriteLine($"parse result      : {result}");
        if (replay == null) return;

        Console.WriteLine($"map               : {replay.Map}  size {replay.MapSize.X}x{replay.MapSize.Y}");
        Console.WriteLine($"length            : {replay.ReplayLength}  ({replay.Frames} frames)");
        Console.WriteLine($"units (all)       : {replay.Units.Count}");
        Console.WriteLine($"tracker events    : {replay.TrackerEvents?.Count}");
        Console.WriteLine($"game events       : {replay.GameEvents?.Count}");
        Console.WriteLine($"objectives        : {replay.TeamObjectives?.Sum(t => t.Count)}");
        Console.WriteLine($"xp breakdown pts  : {replay.TeamPeriodicXPBreakdown?.FirstOrDefault()?.Count}");
        Console.WriteLine();

        // Hero units carry the per-player movement track.
        var heroes = replay.Units.Where(u => u.Group == Unit.UnitGroup.Hero).ToList();
        Console.WriteLine($"hero units        : {heroes.Count}");
        Console.WriteLine();
        // Stitch every life-segment together per player, and separate the positions the
        // replay actually recorded from the ones the parser interpolated between them.
        Console.WriteLine($"{"player",-16}{"all pts",9}{"real",8}{"real dt med",13}{"real dt p90",13}");
        foreach (var g in heroes.Where(h => h.PlayerControlledBy != null)
                                .GroupBy(h => h.PlayerControlledBy.Name).OrderBy(g => g.Key))
        {
            var all = g.SelectMany(h => h.Positions ?? new System.Collections.Generic.List<Position>())
                       .OrderBy(p => p.TimeSpan).ToList();
            var real = all.Where(p => !p.IsEstimated).ToList();
            if (real.Count < 3) { Console.WriteLine($"{g.Key,-16}{all.Count,9}{real.Count,8}"); continue; }
            var d = real.Zip(real.Skip(1), (a, b) => (b.TimeSpan - a.TimeSpan).TotalSeconds)
                        .Where(x => x > 0).OrderBy(x => x).ToList();
            Console.WriteLine($"{g.Key,-16}{all.Count,9}{real.Count,8}{d[d.Count / 2],12:F1}s{d[(int)(d.Count * 0.9)],12:F1}s");
        }

        // Command stream: what kinds of player events are actually in there.
        Console.WriteLine();
        Console.WriteLine("top game-event types:");
        foreach (var g in replay.GameEvents.GroupBy(e => e.eventType).OrderByDescending(g => g.Count()).Take(6))
            Console.WriteLine($"  {g.Key,-28}{g.Count(),8}");

        // Deaths are fully attributed - the backbone of any fight analysis.
        var deaths = replay.Units.Where(u => u.Group == Unit.UnitGroup.Hero && u.TimeSpanDied.HasValue).ToList();
        Console.WriteLine();
        Console.WriteLine($"hero deaths       : {deaths.Count}");
        foreach (var d in deaths.OrderBy(d => d.TimeSpanDied).Take(5))
            Console.WriteLine($"  {d.TimeSpanDied.Value:mm\\:ss}  {d.Name,-14} at ({d.PointDied?.X},{d.PointDied?.Y})  killed by {d.PlayerKilledBy?.Name ?? "?"} ({d.UnitKilledBy?.Name ?? "?"})");
    }
}
