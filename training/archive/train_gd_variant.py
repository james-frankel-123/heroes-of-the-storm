"""Train a single Generic Draft variant. Usage: python train_gd_variant.py <variant_idx>"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch, numpy as np
from shared import load_replay_data, split_data
from train_generic_draft import MODEL_VARIANTS, DraftDataset, train_single_model

idx = int(sys.argv[1])
data = load_replay_data()
train_data, test_data = split_data(data)
train_ds = DraftDataset(train_data)
test_ds = DraftDataset(test_data)
device = torch.device("cuda")
loss = train_single_model(idx, MODEL_VARIANTS[idx], train_ds, test_ds, device)
print(f"DONE variant {idx}: test_loss={loss:.4f}")
