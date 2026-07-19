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
}
