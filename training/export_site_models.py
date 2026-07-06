"""
Export the rerun2026 snapshot-retrained models to public/models/ for browser
inference on the /draft page.

Site models (see src/lib/draft/ai-inference.ts for the consuming code):
  - draft_policy.onnx     <- rerun2026/mcts_runs/J_800sim_s9/draft_policy.pt
        AlphaZero policy/value net. Inputs: state [B,290], valid_mask [B,90].
        Outputs: policy_logits [B,90], value [B].
  - generic_draft_0.onnx  <- rerun2026/models/generic_draft_0.pt
        Opponent model. Inputs: state [B,289], valid_mask [B,90].
        Output: hero_logits [B,90]. (Plus the pre-quantized _int8 variant,
        copied from rerun2026 after a parity check.)
  - win_probability.onnx  <- rerun2026/models/wp_enriched_256.pt
        Enriched WP model (283 = 197 base + 86 enriched features; the site
        computes the same 9 enriched groups in computeEnrichedFeatures).
        Input: input [B,283]. Output: win_probability [B].

Reuses the exact export path of train_draft_policy.py / train_generic_draft.py /
train_win_probability.py (torch.onnx.export -> embed_onnx_weights -> optimize_onnx).

Usage:
    python training/export_site_models.py
"""
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from shared import embed_onnx_weights, optimize_onnx, NUM_HEROES  # noqa: E402
from train_draft_policy import AlphaZeroDraftNet, STATE_DIM  # noqa: E402
from train_generic_draft import GenericDraftModel, INPUT_DIM as GD_INPUT_DIM  # noqa: E402
from sweep_enriched_wp import WinProbEnrichedModel  # noqa: E402

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
RERUN = os.path.join(TRAINING_DIR, "rerun2026")
SITE_MODELS = os.path.join(os.path.dirname(TRAINING_DIR), "public", "models")

POLICY_PT = os.path.join(RERUN, "mcts_runs", "J_800sim_s9", "draft_policy.pt")
GD_PT = os.path.join(RERUN, "models", "generic_draft_0.pt")
GD_ONNX_SRC = os.path.join(RERUN, "models", "generic_draft_0.onnx")
GD_INT8_SRC = os.path.join(RERUN, "models", "generic_draft_0_int8.onnx")
WP_PT = os.path.join(RERUN, "models", "wp_enriched_256.pt")
WP_INPUT_DIM = 283


def _ort_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def export_policy():
    net = AlphaZeroDraftNet(size="base", policy_head_type="linear")
    net.load_state_dict(torch.load(POLICY_PT, weights_only=True, map_location="cpu"))
    net.cpu().eval()

    dummy_x = torch.randn(1, STATE_DIM)
    dummy_mask = torch.ones(1, NUM_HEROES)
    onnx_path = os.path.join(SITE_MODELS, "draft_policy.onnx")
    torch.onnx.export(
        net, (dummy_x, dummy_mask), onnx_path,
        input_names=["state", "valid_mask"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "valid_mask": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
    )
    embed_onnx_weights(onnx_path)
    optimize_onnx(onnx_path)

    # Parity check torch vs onnx
    sess = _ort_session(onnx_path)
    x = torch.randn(4, STATE_DIM)
    x[:, -1] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    mask = torch.ones(4, NUM_HEROES)
    with torch.no_grad():
        t_logits, t_value = net(x, mask)
    o = sess.run(None, {"state": x.numpy(), "valid_mask": mask.numpy()})
    lmax = np.abs(t_logits.numpy() - o[0]).max()
    vmax = np.abs(t_value.numpy() - o[1]).max()
    print(f"policy parity: logits maxdiff={lmax:.2e} value maxdiff={vmax:.2e}")
    assert lmax < 1e-3 and vmax < 1e-4, "policy ONNX does not match torch"
    return onnx_path


def export_gd():
    model = GenericDraftModel()
    model.load_state_dict(torch.load(GD_PT, weights_only=True, map_location="cpu"))
    model.cpu().eval()

    dst = os.path.join(SITE_MODELS, "generic_draft_0.onnx")
    dst_int8 = os.path.join(SITE_MODELS, "generic_draft_0_int8.onnx")
    shutil.copyfile(GD_ONNX_SRC, dst)
    shutil.copyfile(GD_INT8_SRC, dst_int8)

    x = torch.randn(8, GD_INPUT_DIM)
    mask = (torch.rand(8, NUM_HEROES) > 0.2).float()
    mask[:, 0] = 1.0  # keep at least one valid hero
    with torch.no_grad():
        t_logits = model(x, mask)
    sess = _ort_session(dst)
    o = sess.run(None, {"state": x.numpy(), "valid_mask": mask.numpy()})[0]
    maxdiff = np.abs(t_logits.numpy() - o).max()
    print(f"gd parity (float): logits maxdiff={maxdiff:.2e}")
    assert maxdiff < 1e-2, "generic_draft_0.onnx does not match generic_draft_0.pt"

    # int8: check argmax agreement on masked logits
    sess8 = _ort_session(dst_int8)
    o8 = sess8.run(None, {"state": x.numpy(), "valid_mask": mask.numpy()})[0]
    agree = (o.argmax(1) == o8.argmax(1)).mean()
    print(f"gd int8 argmax agreement: {agree:.2f}")
    return dst


def export_wp():
    model = WinProbEnrichedModel(WP_INPUT_DIM, [256, 128])
    model.load_state_dict(torch.load(WP_PT, weights_only=True, map_location="cpu"))
    model.cpu().eval()

    dummy = torch.randn(1, WP_INPUT_DIM)
    onnx_path = os.path.join(SITE_MODELS, "win_probability.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["win_probability"],
        dynamic_axes={"input": {0: "batch"}, "win_probability": {0: "batch"}},
    )
    embed_onnx_weights(onnx_path)
    optimize_onnx(onnx_path)

    sess = _ort_session(onnx_path)
    x = torch.randn(8, WP_INPUT_DIM)
    with torch.no_grad():
        t = model(x)
    o = sess.run(None, {"input": x.numpy()})[0]
    maxdiff = np.abs(t.numpy() - o).max()
    print(f"wp parity: maxdiff={maxdiff:.2e}")
    assert maxdiff < 1e-4, "win_probability.onnx does not match wp_enriched_256.pt"
    return onnx_path


if __name__ == "__main__":
    os.makedirs(SITE_MODELS, exist_ok=True)
    export_policy()
    export_gd()
    export_wp()
    # Stale artifacts from the April-era multi-policy setup
    for stale in ["draft_policy.onnx.data", "draft_policy_f400.onnx", "draft_policy_b200.onnx"]:
        p = os.path.join(SITE_MODELS, stale)
        if os.path.exists(p):
            os.remove(p)
            print(f"removed stale {p}")
    print("done")
