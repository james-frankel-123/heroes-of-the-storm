using HotsFever.DraftEngine.Encoding;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace HotsFever.DraftEngine.Models;

/// <summary>
/// Loads and runs the three ONNX models with the exact input/output names the
/// web engine uses (src/lib/draft/ai-inference.ts):
///   draft_policy.onnx    : { state[1,290], valid_mask[1,90] } -> policy_logits[1,90], value[1,1]
///   generic_draft_0.onnx : { state[1,289], valid_mask[1,90] } -> hero_logits[1,90]
///   win_probability.onnx : { input[1,283] }                   -> win_probability[1,1]
///   partial_wp.onnx      : { features[1,283], step[1] i64 }   -> win_probability[1,1]
///
/// partial_wp is OPTIONAL: it is loaded when the file is present and left null
/// otherwise, so an install missing it degrades to the terminal judge instead of
/// failing to start.
///
/// Execution provider is pinned to CPU for run-to-run stability (per the M1
/// parity plan). A SemaphoreSlim serializes Run() calls, mirroring the web
/// engine's inference mutex.
/// </summary>
public sealed class OnnxSessions : IDisposable
{
    private readonly InferenceSession _policy;
    private readonly InferenceSession _gd;
    private readonly InferenceSession _wp;
    private readonly InferenceSession? _partialWp;
    private readonly SemaphoreSlim _inferLock = new(1, 1);

    public OnnxSessions(string policyPath, string gdPath, string wpPath, string? partialWpPath = null)
    {
        var opts = new SessionOptions();
        // CPU EP only — no GPU providers registered — for deterministic-ish results.
        _policy = new InferenceSession(policyPath, opts);
        _gd = new InferenceSession(gdPath, opts);
        _wp = new InferenceSession(wpPath, opts);
        if (partialWpPath != null && File.Exists(partialWpPath))
            _partialWp = new InferenceSession(partialWpPath, opts);
    }

    /// <summary>True when the partial-draft judge is available.</summary>
    public bool HasPartialWinProbability => _partialWp != null;

    /// <summary>Convenience factory: point at a folder containing the .onnx files.</summary>
    public static OnnxSessions FromDirectory(string modelsDir) => new(
        Path.Combine(modelsDir, "draft_policy.onnx"),
        Path.Combine(modelsDir, "generic_draft_0.onnx"),
        Path.Combine(modelsDir, "win_probability.onnx"),
        Path.Combine(modelsDir, "partial_wp.onnx"));

    public IReadOnlyDictionary<string, NodeMetadata> PolicyInputs => _policy.InputMetadata;
    public IReadOnlyDictionary<string, NodeMetadata> PolicyOutputs => _policy.OutputMetadata;
    public IReadOnlyDictionary<string, NodeMetadata> GdOutputs => _gd.OutputMetadata;
    public IReadOnlyDictionary<string, NodeMetadata> WpOutputs => _wp.OutputMetadata;

    public sealed record PolicyResult(float[] PolicyLogits, float Value);

    public PolicyResult RunPolicy(float[] state, float[] validMask)
    {
        Require(state, StateEncoder.PolicyDims, nameof(state));
        Require(validMask, HeroCatalog.NumHeroes, nameof(validMask));

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("state", ToTensor(state, StateEncoder.PolicyDims)),
            NamedOnnxValue.CreateFromTensor("valid_mask", ToTensor(validMask, HeroCatalog.NumHeroes)),
        };

        _inferLock.Wait();
        try
        {
            using var results = _policy.Run(inputs);
            var logits = results.First(r => r.Name == "policy_logits").AsEnumerable<float>().ToArray();
            var value = results.First(r => r.Name == "value").AsEnumerable<float>().First();
            return new PolicyResult(logits, value);
        }
        finally { _inferLock.Release(); }
    }

    public float[] RunGenericDraft(float[] state, float[] validMask)
    {
        Require(state, StateEncoder.GdDims, nameof(state));
        Require(validMask, HeroCatalog.NumHeroes, nameof(validMask));

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("state", ToTensor(state, StateEncoder.GdDims)),
            NamedOnnxValue.CreateFromTensor("valid_mask", ToTensor(validMask, HeroCatalog.NumHeroes)),
        };

        _inferLock.Wait();
        try
        {
            using var results = _gd.Run(inputs);
            return results.First(r => r.Name == "hero_logits").AsEnumerable<float>().ToArray();
        }
        finally { _inferLock.Release(); }
    }

    public float RunWinProbability(float[] input)
    {
        Require(input, StateEncoder.WpDims, nameof(input));

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", ToTensor(input, StateEncoder.WpDims)),
        };

        _inferLock.Wait();
        try
        {
            using var results = _wp.Run(inputs);
            return results.First(r => r.Name == "win_probability").AsEnumerable<float>().First();
        }
        finally { _inferLock.Release(); }
    }

    /// <summary>
    /// Partial-draft win probability. Same 283 enriched features as the terminal judge
    /// plus a step embedding, so it stays state-sensitive at every stage of the draft
    /// rather than only scoring finished comps. Throws if the model wasn't loaded —
    /// check <see cref="HasPartialWinProbability"/> first.
    /// </summary>
    public float RunPartialWinProbability(float[] input, int step)
    {
        Require(input, StateEncoder.WpDims, nameof(input));
        if (_partialWp == null)
            throw new InvalidOperationException("partial_wp.onnx was not loaded");

        var stepTensor = new DenseTensor<long>(new long[] { Math.Clamp(step, 0, 15) }, new[] { 1 });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("features", ToTensor(input, StateEncoder.WpDims)),
            NamedOnnxValue.CreateFromTensor("step", stepTensor),
        };

        _inferLock.Wait();
        try
        {
            using var results = _partialWp.Run(inputs);
            return results.First(r => r.Name == "win_probability").AsEnumerable<float>().First();
        }
        finally { _inferLock.Release(); }
    }

    private static DenseTensor<float> ToTensor(float[] data, int dim)
        => new(data, new[] { 1, dim });

    private static void Require(float[] v, int len, string name)
    {
        if (v.Length != len)
            throw new ArgumentException($"{name} must be length {len}, got {v.Length}");
    }

    public void Dispose()
    {
        _policy.Dispose();
        _gd.Dispose();
        _wp.Dispose();
        _partialWp?.Dispose();
        _inferLock.Dispose();
    }
}
