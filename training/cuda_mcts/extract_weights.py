"""Extract flattened weight arrays and offset dicts for CUDA kernels."""
import torch
import numpy as np


def extract_policy_weights(model):
    """Extract flattened weights + offsets for AlphaZeroDraftNet.
    Returns (flat_np, offsets_dict) where offsets map 'name' -> int offset.
    """
    tensors = []
    offsets = {}
    offset = 0

    # Parameters first
    for name, param in model.named_parameters():
        offsets[name] = offset
        tensors.append(param.data.cpu().flatten())
        offset += param.numel()

    # Buffers (running_mean, running_var -- skip num_batches_tracked)
    for name, buf in model.named_buffers():
        if 'num_batches_tracked' in name:
            continue
        offsets[name] = offset
        tensors.append(buf.data.cpu().flatten())
        offset += buf.numel()

    flat = torch.cat(tensors).numpy().astype(np.float32)
    return flat, offsets


def extract_gd_weights(model):
    """Extract flattened weights + offsets for GenericDraftModel."""
    tensors = []
    offsets = {}
    offset = 0

    for name, param in model.named_parameters():
        offsets[name] = offset
        tensors.append(param.data.cpu().flatten())
        offset += param.numel()

    flat = torch.cat(tensors).numpy().astype(np.float32)
    return flat, offsets
