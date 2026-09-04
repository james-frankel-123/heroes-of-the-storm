using System.Net.Http;
using System.Reflection;
using System.Text.Json;

namespace HotsFever.Overlay;

/// <summary>
/// What build is this, and is there a newer one?
///
/// Release assets are re-uploaded in place under a fixed tag, so the tag can't be used
/// to detect a new build. The website publishes /draft-coach-version.json instead — a
/// tiny file we control, served by Vercel, with no API rate limit — and the app compares
/// it against the version compiled in here.
/// </summary>
public static class AppVersion
{
    private const string VersionUrl = "https://hotsfever.com/draft-coach-version.json";

    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(10) };

    /// <summary>The running build's version, e.g. "0.2.0".</summary>
    public static string Current { get; } =
        Assembly.GetExecutingAssembly().GetName().Version is Version v
            ? $"{v.Major}.{v.Minor}.{v.Build}"
            : "0.0.0";

    private sealed class Manifest
    {
        public string? version { get; set; }
        public string? url { get; set; }
        public string? notes { get; set; }
    }

    /// <summary>
    /// Returns a short "update available" message, or null when we're current, the
    /// check fails, or the network is unavailable. Never throws: an update check is a
    /// convenience and must not be able to take the overlay down.
    /// </summary>
    public static async Task<string?> CheckForUpdateAsync()
    {
        try
        {
            var json = await Http.GetStringAsync(VersionUrl);
            var manifest = JsonSerializer.Deserialize<Manifest>(json);
            if (manifest?.version is not string latest) return null;
            if (!IsNewer(latest, Current)) return null;

            string notes = string.IsNullOrWhiteSpace(manifest.notes) ? "" : " — " + manifest.notes;
            return $"Version {latest} is available (you have {Current}){notes}. " +
                   $"Download it at {manifest.url ?? "hotsfever.com/draft-coach"} and run the installer.";
        }
        catch
        {
            return null;
        }
    }

    /// <summary>Numeric compare of dotted versions, so 0.10.0 beats 0.9.0.</summary>
    internal static bool IsNewer(string candidate, string current)
    {
        static int[] Parts(string s) => s.Split('.')
            .Select(p => int.TryParse(p, out int n) ? n : 0)
            .Concat(new[] { 0, 0, 0 })
            .Take(3)
            .ToArray();

        var a = Parts(candidate);
        var b = Parts(current);
        for (int i = 0; i < 3; i++)
        {
            if (a[i] > b[i]) return true;
            if (a[i] < b[i]) return false;
        }
        return false;
    }
}
