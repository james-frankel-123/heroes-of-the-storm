# Parity Oracle (model layer)

Validates the native C# draft engine against **onnxruntime-web** — the runtime the
website actually uses — so we can be confident the C# port is equivalent *within
margin*, not just internally consistent.

## What it does

`gen-golden.mjs` builds 100 diverse draft states, encodes them, runs all three
ONNX models through onnxruntime-web (pinned to the same version as the C# ORT),
and writes `oracle-golden.json` — reference input tensors + model outputs.

The C# side (`native/.../OracleParityTests.cs`) then:

- **Layer 1 (encoding):** re-encodes each case with `StateEncoder` and asserts the
  input tensors match the web engine's byte-for-byte.
- **Layer 2 (runtime):** runs the same inputs through `Microsoft.ML.OnnxRuntime`
  and asserts the outputs match onnxruntime-web within a slim tolerance, and that
  the top pick (argmax over available heroes) agrees on every case.

Observed divergence is ~`1e-6` (tolerance is `5e-3`); argmax agrees 100%.

## Regenerating

Run this after any model change (`public/models/*.onnx`) or encoding change, then
commit the updated `oracle-golden.json`:

```
cd tools/oracle
npm install     # first time only
npm run gen
```

Deterministic: a seeded PRNG drives case generation and the model weights are
fixed, so regeneration produces an identical file unless the models changed.

## Note on the behavioral (MCTS) layer

This oracle covers the **model math** (the CTO's "precompute pairs, validate within
margin" approach). Validating the full **MCTS search** end-to-end against the live
web engine additionally requires a mock-RNG / fixed-sim mode in the TypeScript
engine (a small, behavior-neutral refactor of `src/lib/draft/mcts-search.ts`) so
both sides step through the same random draws. That is the next oracle increment.
