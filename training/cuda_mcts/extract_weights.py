"""Extract flattened weight arrays and offset dicts for CUDA kernels."""
import torch
import torch.nn as nn
import numpy as np


def extract_policy_weights(model):
    """Extract flattened weights + offsets for AlphaZeroDraftNet.
    Returns (flat_np, offsets_dict) where offsets map 'name' -> int offset.
    Includes architecture dims (hdim, cdim, edim, n_blocks) for the kernel.
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

    # Add architecture dimensions for kernel
    offsets['hdim'] = getattr(model, '_hdim', 768)
    offsets['edim'] = getattr(model, '_edim', 256)
    if hasattr(model, 'res_blocks'):
        offsets['n_blocks'] = len(model.res_blocks)
    else:
        offsets['n_blocks'] = 3
    offsets['cdim'] = model.compress1.out_features

    # Policy head type and layer offsets
    pht = getattr(model, '_policy_head_type', 'linear')
    type_map = {'linear': 0, 'deep': 1, 'step': 2, 'deep_step': 3}
    offsets['policy_head_type'] = type_map.get(pht, 0)

    if pht == 'linear':
        # Legacy: single linear layer, offsets already in 'policy_head.weight'/'policy_head.bias'
        offsets['policy_n_layers'] = 1
        offsets['step_embed_w'] = -1
    else:
        # Deep/step heads: policy_head is nn.Sequential
        layers = []
        ph = model.policy_head
        for name_i, mod in ph.named_modules():
            if isinstance(mod, nn.Linear):
                seq_idx = int(name_i)
                layers.append({
                    'in': mod.in_features, 'out': mod.out_features,
                    'w': offsets[f'policy_head.{seq_idx}.weight'],
                    'b': offsets[f'policy_head.{seq_idx}.bias'],
                })
        offsets['policy_n_layers'] = len(layers)
        offsets['policy_layers'] = layers

        if hasattr(model, 'step_embed'):
            offsets['step_embed_w'] = offsets['step_embed.weight']
        else:
            offsets['step_embed_w'] = -1

        # Legacy compat: point to first layer
        offsets['policy_head.weight'] = layers[0]['w']
        offsets['policy_head.bias'] = layers[0]['b']

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


def extract_wp_weights(model):
    """Extract flattened weights + offsets for WinProbEnrichedModel.
    Returns (flat_np, name_to_offset).
    """
    tensors = []
    offsets = {}
    offset = 0

    for name, param in model.named_parameters():
        offsets[name] = offset
        tensors.append(param.data.cpu().flatten())
        offset += param.numel()

    for name, buf in model.named_buffers():
        if 'num_batches_tracked' in name:
            continue
        offsets[name] = offset
        tensors.append(buf.data.cpu().flatten())
        offset += buf.numel()

    flat = torch.cat(tensors).numpy().astype(np.float32)
    return flat, offsets


def build_wp_net_offsets(model, name_to_offset, input_dim):
    """Build WPNetOffsets config dict for WinProbEnrichedModel.
    The dict maps directly to the WPNetOffsets C struct fields.
    """
    modules = list(model.net)
    layers = []

    i = 0
    while i < len(modules):
        m = modules[i]
        if isinstance(m, nn.Linear):
            has_bn = (i + 1 < len(modules) and isinstance(modules[i + 1], nn.BatchNorm1d))
            # ReLU follows BN (or directly follows Linear if no BN)
            relu_check_idx = (i + 2) if has_bn else (i + 1)
            has_relu = (relu_check_idx < len(modules) and isinstance(modules[relu_check_idx], nn.ReLU))
            layers.append({
                'seq_idx': i,
                'in': m.in_features,
                'out': m.out_features,
                'has_bn': has_bn,
                'bn_idx': (i + 1) if has_bn else None,
                'use_relu': has_relu,
            })
        i += 1

    num_layers = len(layers)
    offsets = {
        'num_layers': num_layers,
        'input_dim': input_dim,
        'use_sigmoid': 1,
        'layer_in': [0] * 6,
        'layer_out': [0] * 6,
        'weight_off': [0] * 6,
        'bias_off': [0] * 6,
        'has_bn': [0] * 6,
        'bn_w_off': [0] * 6,
        'bn_b_off': [0] * 6,
        'bn_m_off': [0] * 6,
        'bn_v_off': [0] * 6,
        'use_relu': [0] * 6,
    }

    for l, layer in enumerate(layers):
        si = layer['seq_idx']
        offsets['layer_in'][l] = layer['in']
        offsets['layer_out'][l] = layer['out']
        offsets['weight_off'][l] = name_to_offset[f'net.{si}.weight']
        offsets['bias_off'][l] = name_to_offset[f'net.{si}.bias']
        offsets['has_bn'][l] = 1 if layer['has_bn'] else 0
        offsets['use_relu'][l] = 1 if layer['use_relu'] else 0

        if layer['has_bn']:
            bi = layer['bn_idx']
            offsets['bn_w_off'][l] = name_to_offset[f'net.{bi}.weight']
            offsets['bn_b_off'][l] = name_to_offset[f'net.{bi}.bias']
            offsets['bn_m_off'][l] = name_to_offset[f'net.{bi}.running_mean']
            offsets['bn_v_off'][l] = name_to_offset[f'net.{bi}.running_var']

    return offsets


def extract_lookup_tables(stats_cache, step_embed_weights=None):
    """Convert StatsCache to flat byte blob matching WPLookupTables C struct layout.
    Returns numpy uint8 array that can be memcpy'd directly into the GPU struct.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from shared import (HEROES, HERO_TO_IDX, MAPS, MAP_TO_IDX,
                        SKILL_TIERS, TIER_TO_IDX,
                        HERO_ROLE_FINE, FINE_ROLE_TO_IDX)

    NUM_HEROES = 90
    NUM_MAPS = 14
    NUM_TIERS = 3
    MAX_COMP_ENTRIES = 512

    # hero_wr: float[3][90]
    hero_wr = np.full((NUM_TIERS, NUM_HEROES), 50.0, dtype=np.float32)
    for tier_name, tier_data in stats_cache.hero_wr.items():
        ti = TIER_TO_IDX.get(tier_name)
        if ti is None:
            continue
        for hero, wr in tier_data.items():
            hi = HERO_TO_IDX.get(hero)
            if hi is not None:
                hero_wr[ti, hi] = wr

    # hero_map_wr: float[3][14][90] — 0 means "use hero_wr fallback"
    hero_map_wr = np.zeros((NUM_TIERS, NUM_MAPS, NUM_HEROES), dtype=np.float32)
    for tier_name, maps_data in stats_cache.hero_map_wr.items():
        ti = TIER_TO_IDX.get(tier_name)
        if ti is None:
            continue
        for map_name, heroes_data in maps_data.items():
            mi = MAP_TO_IDX.get(map_name)
            if mi is None:
                continue
            for hero, (wr, games) in heroes_data.items():
                hi = HERO_TO_IDX.get(hero)
                if hi is not None and games >= 50:
                    hero_map_wr[ti, mi, hi] = wr

    # pairwise_counter: float[3][90][90] — 0 means no data
    pairwise_counter = np.zeros((NUM_TIERS, NUM_HEROES, NUM_HEROES), dtype=np.float32)
    for tier_name, rel_data in stats_cache.pairwise.items():
        ti = TIER_TO_IDX.get(tier_name)
        if ti is None:
            continue
        for ha, opponents in rel_data.get("against", {}).items():
            hai = HERO_TO_IDX.get(ha)
            if hai is None:
                continue
            for hb, (wr, games) in opponents.items():
                hbi = HERO_TO_IDX.get(hb)
                if hbi is not None and games >= 30:
                    pairwise_counter[ti, hai, hbi] = wr

    # pairwise_synergy: float[3][90][90]
    pairwise_synergy = np.zeros((NUM_TIERS, NUM_HEROES, NUM_HEROES), dtype=np.float32)
    for tier_name, rel_data in stats_cache.pairwise.items():
        ti = TIER_TO_IDX.get(tier_name)
        if ti is None:
            continue
        for ha, partners in rel_data.get("with", {}).items():
            hai = HERO_TO_IDX.get(ha)
            if hai is None:
                continue
            for hb, (wr, games) in partners.items():
                hbi = HERO_TO_IDX.get(hb)
                if hbi is not None and games >= 30:
                    pairwise_synergy[ti, hai, hbi] = wr

    # hero_meta: float[3][90][2]
    hero_meta = np.zeros((NUM_TIERS, NUM_HEROES, 2), dtype=np.float32)
    for tier_name, meta_data in stats_cache.hero_meta.items():
        ti = TIER_TO_IDX.get(tier_name)
        if ti is None:
            continue
        for hero, (pr, br) in meta_data.items():
            hi = HERO_TO_IDX.get(hero)
            if hi is not None:
                hero_meta[ti, hi, 0] = pr
                hero_meta[ti, hi, 1] = br

    # hero_fine_role: int[90]
    hero_fine_role = np.zeros(NUM_HEROES, dtype=np.int32)
    for hero, role in HERO_ROLE_FINE.items():
        hi = HERO_TO_IDX.get(hero)
        ri = FINE_ROLE_TO_IDX.get(role, 0)
        if hi is not None:
            hero_fine_role[hi] = ri

    # hero_blizz_role: int[90] — map fine→blizz for comp_wr lookup
    BLIZZ_ROLES = ["Bruiser", "Healer", "Melee Assassin", "Ranged Assassin", "Support", "Tank"]
    BLIZZ_ROLE_TO_IDX = {r: i for i, r in enumerate(BLIZZ_ROLES)}
    FINE_TO_BLIZZ = {
        "tank": "Tank", "bruiser": "Bruiser", "healer": "Healer",
        "ranged_aa": "Ranged Assassin", "ranged_mage": "Ranged Assassin",
        "melee_assassin": "Melee Assassin", "support_utility": "Support",
        "varian": "Bruiser", "pusher": "Ranged Assassin",
    }
    hero_blizz_role = np.zeros(NUM_HEROES, dtype=np.int32)
    for hero, fine_role in HERO_ROLE_FINE.items():
        hi = HERO_TO_IDX.get(hero)
        blizz = FINE_TO_BLIZZ.get(fine_role, "Ranged Assassin")
        bi = BLIZZ_ROLE_TO_IDX.get(blizz, 3)
        if hi is not None:
            hero_blizz_role[hi] = bi

    # comp data: comp_keys/wr/games/count
    comp_keys = np.zeros((NUM_TIERS, MAX_COMP_ENTRIES), dtype=np.int32)
    comp_wr_arr = np.zeros((NUM_TIERS, MAX_COMP_ENTRIES), dtype=np.float32)
    comp_games_arr = np.zeros((NUM_TIERS, MAX_COMP_ENTRIES), dtype=np.float32)
    comp_count = np.zeros(NUM_TIERS, dtype=np.int32)

    for tier_name, tier_comps in stats_cache.comp_data.items():
        ti = TIER_TO_IDX.get(tier_name)
        if ti is None:
            continue
        count = 0
        for role_key, (wr, games) in tier_comps.items():
            if count >= MAX_COMP_ENTRIES:
                break
            roles = role_key.split(",")
            role_indices = sorted([BLIZZ_ROLE_TO_IDX.get(r.strip(), 3) for r in roles])
            packed = 0
            for j, ri in enumerate(role_indices):
                packed |= (ri << (j * 4))
            comp_keys[ti, count] = packed
            comp_wr_arr[ti, count] = wr
            comp_games_arr[ti, count] = games
            count += 1
        comp_count[ti] = count

    # Concatenate matching C struct layout (all fields contiguous, no padding)
    # Step embedding for partial WP model (16 steps × 8 dims)
    if step_embed_weights is not None:
        step_embed = np.array(step_embed_weights, dtype=np.float32).reshape(16, 8)
    else:
        step_embed = np.zeros((16, 8), dtype=np.float32)

    blob = b''.join([
        hero_wr.tobytes(),
        hero_map_wr.tobytes(),
        pairwise_counter.tobytes(),
        pairwise_synergy.tobytes(),
        hero_meta.tobytes(),
        hero_fine_role.tobytes(),
        hero_blizz_role.tobytes(),
        comp_keys.tobytes(),
        comp_wr_arr.tobytes(),
        comp_games_arr.tobytes(),
        comp_count.tobytes(),
        step_embed.tobytes(),
    ])
    print(f"  LUT blob: {len(blob)} bytes ({len(blob)/1024:.1f} KB)")
    return np.frombuffer(blob, dtype=np.uint8).copy()
