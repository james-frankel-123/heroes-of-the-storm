"""Export the production partial-WP checkpoint to public/models/partial_wp.onnx
with the site export conventions (embedded weights, optimizer pass, parity
assert). Inputs: features [B,283] float32, step [B] int64 -> win_probability [B].

Usage: python3 training/production_refresh/export_partial_wp.py
"""
import os
import sys

import numpy as np
import torch

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
from shared import embed_onnx_weights, optimize_onnx  # noqa: E402
from train_partial_wp import PartialStateWP  # noqa: E402

CKPT = os.environ.get(
    "PARTIAL_WP_CKPT",
    os.path.join(TRAINING, "production_refresh", "2026-07-14", "partial_wp_prod.pt"))
OUT = os.path.join(os.path.dirname(TRAINING), "public", "models", "partial_wp.onnx")


def main():
    ckpt = torch.load(CKPT, weights_only=True, map_location="cpu")
    model = PartialStateWP(input_dim=ckpt.get("input_dim", 283),
                           step_embed_dim=ckpt.get("step_embed_dim", 8),
                           hidden=tuple(ckpt.get("hidden", (256, 128))))
    model.load_state_dict(ckpt["model_state_dict"])
    model.cpu().eval()
    print(f"checkpoint: test acc {ckpt.get('best_test_acc')}")

    feats = torch.randn(1, 283)
    step = torch.tensor([7], dtype=torch.long)
    torch.onnx.export(
        model, (feats, step), OUT,
        input_names=["features", "step"],
        output_names=["win_probability"],
        dynamic_axes={"features": {0: "batch"}, "step": {0: "batch"},
                      "win_probability": {0: "batch"}},
    )
    embed_onnx_weights(OUT)
    optimize_onnx(OUT)

    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    x = torch.randn(8, 283)
    s = torch.randint(0, 16, (8,), dtype=torch.long)
    with torch.no_grad():
        t_out = model(x, s).numpy()
    o = sess.run(None, {"features": x.numpy(), "step": s.numpy()})[0].reshape(-1)
    maxdiff = np.abs(t_out - o).max()
    print(f"partial_wp parity: maxdiff={maxdiff:.2e}")
    assert maxdiff < 1e-4, "partial_wp.onnx does not match checkpoint"
    print(f"exported {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
