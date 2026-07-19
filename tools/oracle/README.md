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

## Behavioral (MCTS) layer

`gen-mcts-golden.ts` (run: `npm run gen:mcts`, via tsx) runs the **real** TypeScript MCTS
(`src/lib/draft/mcts-search.ts`) through onnxruntime-web, driven by a shared
mock-RNG sequence + fixed sim count (uncapped time). It writes `mcts-golden.json`
(the RNG sequence + per-case recommendations).

The C# side (`MctsBehavioralParityTests.cs`) runs its search through the same RNG
sequence and compares. Because the shared RNG makes the searches deterministic and
the ~1e-6 model divergence rarely reaches a branch boundary, agreement is in
practice **exact** (observed: top-1 20/20, exact top-5 20/20, value diff 0.00000),
though the test only asserts statistical thresholds to stay robust.

This relies on `runMCTSSearch` accepting an optional `options.rng` / sim overrides
— a behavior-neutral addition; the defaults reproduce production exactly.
