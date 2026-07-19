namespace HotsFever.DraftEngine.Tests;

internal static class TestPaths
{
    /// <summary>
    /// Walk up from the test assembly to the repo root and return public/models,
    /// where the three .onnx files live.
    /// </summary>
    public static string ModelsDir()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, "public", "models");
            if (File.Exists(Path.Combine(candidate, "draft_policy.onnx"))) return candidate;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException(
            "Could not locate public/models (draft_policy.onnx) above " + AppContext.BaseDirectory);
    }

    /// <summary>Walk up to the repo root and return src/lib/data, where the stat-table JSON lives.</summary>
    public static string DataDir()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, "src", "lib", "data");
            if (File.Exists(Path.Combine(candidate, "draft-stats-decayed.json"))) return candidate;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException(
            "Could not locate src/lib/data (draft-stats-decayed.json) above " + AppContext.BaseDirectory);
    }

    public static string StatsJson() => Path.Combine(DataDir(), "draft-stats-decayed.json");
    public static string CompositionsJson() => Path.Combine(DataDir(), "compositions.json");

    /// <summary>Path to the parity oracle's golden file (tools/oracle/oracle-golden.json), or null if not generated.</summary>
    public static string? OracleGolden() => FindUp(Path.Combine("tools", "oracle", "oracle-golden.json"));

    /// <summary>Path to the MCTS behavioral golden file (tools/oracle/mcts-golden.json), or null if not generated.</summary>
    public static string? MctsGolden() => FindUp(Path.Combine("tools", "oracle", "mcts-golden.json"));

    private static string? FindUp(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return null;
    }
}
