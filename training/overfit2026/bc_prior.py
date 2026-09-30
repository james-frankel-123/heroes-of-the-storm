"""
Outcome-free search prior: distill the paper's GD behavioral-cloning model
(rerun2026/models/generic_draft_0.pt, trained on human picks only, never on
win/loss) into the AlphaZeroDraftNet policy head, so the CUDA kernel can use
it as a PUCT prior. States come from the phase0 GD cache (human draft
states). Loss: cross-entropy to GD's masked softmax.

Output: models/bc_prior.pt
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch
import torch.nn.functional as F


def main(n_samples=4_000_000, epochs=3, bs=4096):
    from rerun2026 import common
    from train_generic_draft import GenericDraftModel
    from train_draft_policy import AlphaZeroDraftNet
    dev = torch.device("cuda")
    mm = common.memmap_open(os.path.join(common.CACHE_DIR, "gd_train"))
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(mm["meta"]["n"], n_samples, replace=False))
    S = torch.tensor(np.asarray(mm["states"][idx]), dtype=torch.float32)
    occ = S[:, :270].reshape(-1, 3, 90).sum(1)
    M = (1.0 - occ.clamp(0, 1))
    gd = GenericDraftModel()
    gd.load_state_dict(torch.load(os.path.join(common.MODELS_DIR, "generic_draft_0.pt"),
                                  weights_only=True, map_location="cpu"))
    gd.eval().to(dev)
    net = AlphaZeroDraftNet(size="base", policy_head_type="linear").to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(S)
    hold = slice(n - 50000, n)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n - 50000)
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x, m = S[b].to(dev), M[b].to(dev)
            with torch.no_grad():
                tgt = F.softmax(gd(x, m), -1)
            x290 = torch.cat([x, torch.zeros(len(x), 1, device=dev)], 1)
            logits, _ = net(x290, m)
            loss = -(tgt * F.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            x, m = S[hold].to(dev), M[hold].to(dev)
            tgt = F.softmax(gd(x, m), -1)
            lg, _ = net(torch.cat([x, torch.zeros(len(x), 1, device=dev)], 1), m)
            lp = F.log_softmax(lg, -1)
            ce = -(tgt * lp).sum(-1).mean().item()
            ent = -(tgt * torch.log(tgt.clamp_min(1e-12))).sum(-1).mean().item()
            agree = (lg.argmax(-1) == tgt.argmax(-1)).float().mean().item()
        print(f"epoch {ep + 1}: train CE {tot / len(perm):.4f}  holdout CE {ce:.4f} "
              f"(GD entropy {ent:.4f}, KL {ce - ent:.4f})  argmax agree {agree:.3f}", flush=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()},
               os.path.join(HERE, "models", "bc_prior.pt"))


if __name__ == "__main__":
    main()
