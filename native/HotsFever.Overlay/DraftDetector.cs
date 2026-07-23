using OpenCvSharp;

namespace HotsFever.Overlay;

/// <summary>
/// Decides whether a captured frame is the HotS draft screen, by template-
/// matching a stable, draft-only, center-anchored UI element ("Browse All
/// Heroes") against the frame. Frames are normalized to a reference height so
/// the same template works across resolutions; aspect-ratio differences don't
/// matter because MatchTemplate searches the whole frame.
///
/// M4 increment 2a: the draft-start signal that drives the capture loop. Hero
/// recognition per slot comes next.
/// </summary>
public static class DraftDetector
{
    private const int RefHeight = 1440;      // template authored at 1440p
    private const double MatchThreshold = 0.70;

    private static Mat? _template;

    private static Mat Template()
    {
        if (_template == null)
        {
            var path = System.IO.Path.Combine(AppContext.BaseDirectory, "Assets", "draft", "browse-heroes.png");
            _template = Cv2.ImRead(path, ImreadModes.Grayscale);
        }
        return _template;
    }

    /// <summary>Best template-match score in [0,1] for the "draft screen" signal.</summary>
    public static double Score(byte[] bgra, int width, int height)
    {
        if (width <= 0 || height <= 0) return 0;
        using var frame = Mat.FromPixelData(height, width, MatType.CV_8UC4, bgra);
        using var gray = new Mat();
        Cv2.CvtColor(frame, gray, ColorConversionCodes.BGRA2GRAY);

        // Normalize to the reference height so the template scale matches.
        double scale = (double)RefHeight / height;
        using var scaled = new Mat();
        Cv2.Resize(gray, scaled, new Size(Math.Max(1, (int)Math.Round(width * scale)), RefHeight));

        var tmpl = Template();
        if (tmpl.Empty() || scaled.Rows < tmpl.Rows || scaled.Cols < tmpl.Cols) return 0;

        using var result = new Mat();
        Cv2.MatchTemplate(scaled, tmpl, result, TemplateMatchModes.CCoeffNormed);
        Cv2.MinMaxLoc(result, out _, out double max);
        return max;
    }

    public static bool IsDraftScreen(byte[] bgra, int width, int height)
        => Score(bgra, width, height) >= MatchThreshold;

    /// <summary>Score a saved PNG (for offline verification against known frames).</summary>
    public static double ScoreFile(string path)
    {
        using var frame = Cv2.ImRead(path, ImreadModes.Grayscale);
        if (frame.Empty()) return -1;
        double scale = (double)RefHeight / frame.Rows;
        using var scaled = new Mat();
        Cv2.Resize(frame, scaled, new Size(Math.Max(1, (int)Math.Round(frame.Cols * scale)), RefHeight));
        var tmpl = Template();
        if (tmpl.Empty() || scaled.Rows < tmpl.Rows || scaled.Cols < tmpl.Cols) return -2;
        using var result = new Mat();
        Cv2.MatchTemplate(scaled, tmpl, result, TemplateMatchModes.CCoeffNormed);
        Cv2.MinMaxLoc(result, out _, out double max);
        return max;
    }
}
