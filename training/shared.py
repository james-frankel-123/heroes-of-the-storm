"""
Shared utilities for training pipelines.
Hero encoding, data loading from Postgres, map/tier encoding.
"""
import os
import json
import numpy as np

# 90 heroes sorted alphabetically — must match src/lib/data/hero-roles.ts
HEROES = [
    "Abathur","Alarak","Alexstrasza","Ana","Anduin","Anub'arak","Artanis",
    "Arthas","Auriel","Azmodan","Blaze","Brightwing","Cassia","Chen","Cho",
    "Chromie","D.Va","Deathwing","Deckard","Dehaka","Diablo","E.T.C.",
    "Falstad","Fenix","Gall","Garrosh","Gazlowe","Genji","Greymane",
    "Gul'dan","Hanzo","Hogger","Illidan","Imperius","Jaina","Johanna",
    "Junkrat","Kael'thas","Kel'Thuzad","Kerrigan","Kharazim","Leoric",
    "Li Li","Li-Ming","Lt. Morales","Lunara","Lúcio","Maiev","Mal'Ganis",
    "Malfurion","Malthael","Medivh","Mei","Mephisto","Muradin","Murky",
    "Nazeebo","Nova","Orphea","Probius","Qhira","Ragnaros","Raynor",
    "Rehgar","Rexxar","Samuro","Sgt. Hammer","Sonya","Stitches","Stukov",
    "Sylvanas","Tassadar","The Butcher","The Lost Vikings","Thrall","Tracer",
    "Tychus","Tyrael","Tyrande","Uther","Valeera","Valla","Varian",
    "Whitemane","Xul","Yrel","Zagara","Zarya","Zeratul","Zul'jin",
]

NUM_HEROES = len(HEROES)
HERO_TO_IDX = {h: i for i, h in enumerate(HEROES)}

MAPS = [
    "Alterac Pass", "Battlefield of Eternity", "Blackheart's Bay",
    "Braxis Holdout", "Cursed Hollow", "Dragon Shire",
    "Garden of Terror", "Hanamura Temple", "Infernal Shrines",
    "Sky Temple", "Tomb of the Spider Queen", "Towers of Doom",
    "Volskaya Foundry", "Warhead Junction",
]
NUM_MAPS = len(MAPS)
MAP_TO_IDX = {m: i for i, m in enumerate(MAPS)}

SKILL_TIERS = ["low", "mid", "high"]
TIER_TO_IDX = {t: i for i, t in enumerate(SKILL_TIERS)}
NUM_TIERS = len(SKILL_TIERS)


# Fine-grained role mapping for enriched WP model features.
# 8 categories: tank, bruiser, healer, ranged_aa, ranged_mage, melee_assassin, support_utility, varian
# Ranged Assassin split: AA-based (auto-attack primary) vs mage (ability-based)
HERO_ROLE_FINE = {
    # Tanks
    "Anub'arak": "tank", "Arthas": "tank", "Blaze": "tank", "Cho": "tank",
    "Diablo": "tank", "E.T.C.": "tank", "Garrosh": "tank", "Johanna": "tank",
    "Mal'Ganis": "tank", "Mei": "tank", "Muradin": "tank", "Stitches": "tank",
    "Tyrael": "tank",
    # Bruisers
    "Artanis": "bruiser", "Chen": "bruiser", "Deathwing": "bruiser",
    "Dehaka": "bruiser", "D.Va": "bruiser", "Gazlowe": "bruiser",
    "Hogger": "bruiser", "Imperius": "bruiser", "Leoric": "bruiser",
    "Malthael": "bruiser", "Ragnaros": "bruiser", "Rexxar": "bruiser",
    "Sonya": "bruiser", "Thrall": "bruiser", "Xul": "bruiser", "Yrel": "bruiser",
    # Healers
    "Alexstrasza": "healer", "Ana": "healer", "Anduin": "healer",
    "Auriel": "healer", "Brightwing": "healer", "Deckard": "healer",
    "Kharazim": "healer", "Li Li": "healer", "Lt. Morales": "healer",
    "Lúcio": "healer", "Malfurion": "healer", "Rehgar": "healer",
    "Stukov": "healer", "Tyrande": "healer", "Uther": "healer",
    "Whitemane": "healer",
    # Ranged Assassins — AA-based (primary damage from auto-attacks)
    "Cassia": "ranged_aa", "Falstad": "ranged_aa", "Fenix": "ranged_aa",
    "Greymane": "ranged_aa", "Hanzo": "ranged_aa", "Lunara": "ranged_aa",
    "Raynor": "ranged_aa", "Sgt. Hammer": "ranged_aa", "Sylvanas": "ranged_aa",
    "Tracer": "ranged_aa", "Tychus": "ranged_aa", "Valla": "ranged_aa",
    "Zul'jin": "ranged_aa",
    # Ranged Assassins — Mage (primary damage from abilities)
    "Chromie": "ranged_mage", "Gall": "ranged_mage",
    "Genji": "ranged_mage", "Gul'dan": "ranged_mage", "Jaina": "ranged_mage",
    "Junkrat": "ranged_mage", "Kael'thas": "ranged_mage",
    "Kel'Thuzad": "ranged_mage", "Li-Ming": "ranged_mage",
    "Mephisto": "ranged_mage", "Nova": "ranged_mage",
    "Orphea": "ranged_mage", "Probius": "ranged_mage",
    "Tassadar": "ranged_mage",
    # Pushers / Specialists (macro-focused heroes)
    "Azmodan": "pusher", "Nazeebo": "pusher", "Zagara": "pusher",
    "Murky": "pusher", "The Lost Vikings": "pusher",
    # Melee Assassins
    "Alarak": "melee_assassin", "Illidan": "melee_assassin",
    "Kerrigan": "melee_assassin", "Maiev": "melee_assassin",
    "Qhira": "melee_assassin",
    "Samuro": "melee_assassin", "The Butcher": "melee_assassin",
    "Valeera": "melee_assassin", "Zeratul": "melee_assassin",
    # Support / Utility
    "Abathur": "support_utility", "Medivh": "support_utility",
    "Zarya": "support_utility",
    # Varian — own category (can be tank, bruiser, or assassin)
    "Varian": "varian",
}

FINE_ROLE_NAMES = ["tank", "bruiser", "healer", "ranged_aa", "ranged_mage",
                   "melee_assassin", "support_utility", "varian", "pusher"]
FINE_ROLE_TO_IDX = {r: i for i, r in enumerate(FINE_ROLE_NAMES)}

# Two-lane maps
TWO_LANE_MAPS = {"Battlefield of Eternity", "Braxis Holdout", "Hanamura Temple"}


def heroes_to_multi_hot(hero_names: list[str]) -> np.ndarray:
    """Convert a list of hero names to a multi-hot vector of length NUM_HEROES."""
    vec = np.zeros(NUM_HEROES, dtype=np.float32)
    for name in hero_names:
        idx = HERO_TO_IDX.get(name)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def map_to_one_hot(map_name: str) -> np.ndarray:
    """Convert map name to one-hot vector."""
    vec = np.zeros(NUM_MAPS, dtype=np.float32)
    idx = MAP_TO_IDX.get(map_name)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def tier_to_one_hot(tier: str) -> np.ndarray:
    """Convert skill tier to one-hot vector."""
    vec = np.zeros(NUM_TIERS, dtype=np.float32)
    idx = TIER_TO_IDX.get(tier)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def load_replay_data(limit: int | None = None) -> list[dict]:
    """Load replay draft data from Postgres."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError("pip install psycopg2-binary")

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable required")

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    query = """
        SELECT replay_id, game_map, skill_tier, draft_order,
               team0_heroes, team1_heroes, team0_bans, team1_bans, winner
        FROM replay_draft_data
        ORDER BY replay_id
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query)
    columns = [desc[0] for desc in cur.description]
    rows = []
    for row in cur.fetchall():
        d = dict(zip(columns, row))
        # Parse JSON fields if they're strings
        for field in ["draft_order", "team0_heroes", "team1_heroes", "team0_bans", "team1_bans"]:
            if isinstance(d[field], str):
                d[field] = json.loads(d[field])
        rows.append(d)

    cur.close()
    conn.close()
    return rows


def embed_onnx_weights(onnx_path: str):
    """Re-save an ONNX model with all weights embedded (no external .data file)."""
    import onnx
    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(model, onnx_path, save_as_external_data=False)
    # Clean up any leftover .data file
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)


def split_data(data: list, test_frac: float = 0.02, seed: int = 42):
    """Split data into train and test sets."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(data))
    test_size = max(1, int(len(data) * test_frac))
    test_indices = set(indices[:test_size])
    train = [data[i] for i in range(len(data)) if i not in test_indices]
    test = [data[i] for i in range(len(data)) if i in test_indices]
    return train, test


def optimize_onnx(onnx_path: str):
    """Run ONNX Runtime graph optimization (constant folding, fusion, etc.)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = onnx_path  # overwrite in place
    # Creating the session triggers optimization and saves
    ort.InferenceSession(onnx_path, opts)
    print(f"  Optimized: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")


def quantize_onnx(
    onnx_path: str,
    calibration_data: list[dict],
    model_type: str = "policy",
):
    """
    INT8 static quantization with MinMax calibration.
    model_type: "policy" (state+mask→logits+value), "gd" (state+mask→logits),
                or "wp" (input→wp)
    """
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_static, CalibrationMethod
    from onnxruntime.quantization import CalibrationDataReader

    quant_path = onnx_path.replace(".onnx", "_int8.onnx")

    class DraftCalibrationReader(CalibrationDataReader):
        def __init__(self, data, model_type):
            self.data = data
            self.model_type = model_type
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.data):
                return None
            d = self.data[self.idx]
            self.idx += 1

            if self.model_type == "wp":
                # WP model: single input tensor
                t0 = heroes_to_multi_hot(d["team0_heroes"])
                t1 = heroes_to_multi_hot(d["team1_heroes"])
                m = map_to_one_hot(d["game_map"])
                t = tier_to_one_hot(d["skill_tier"])
                x = np.concatenate([t0, t1, m, t]).reshape(1, -1).astype(np.float32)
                return {"input": x}
            else:
                # Policy or GD: state + valid_mask
                t0 = heroes_to_multi_hot(d.get("team0_heroes", []))
                t1 = heroes_to_multi_hot(d.get("team1_heroes", []))
                bans = np.zeros(NUM_HEROES, dtype=np.float32)
                for h in d.get("team0_bans", []) + d.get("team1_bans", []):
                    idx = HERO_TO_IDX.get(h)
                    if idx is not None:
                        bans[idx] = 1.0
                m = map_to_one_hot(d["game_map"])
                t = tier_to_one_hot(d["skill_tier"])
                state = np.concatenate([t0, t1, bans, m, t, [1.0, 1.0]]).reshape(1, -1).astype(np.float32)
                mask = np.ones((1, NUM_HEROES), dtype=np.float32)
                taken = set(d.get("team0_heroes", []) + d.get("team1_heroes", []) +
                           d.get("team0_bans", []) + d.get("team1_bans", []))
                for h in taken:
                    idx = HERO_TO_IDX.get(h)
                    if idx is not None:
                        mask[0, idx] = 0.0
                return {"state": state, "valid_mask": mask}

    reader = DraftCalibrationReader(calibration_data, model_type)

    quantize_static(
        onnx_path,
        quant_path,
        reader,
        calibrate_method=CalibrationMethod.MinMax,
    )

    print(f"  Quantized: {quant_path} ({os.path.getsize(quant_path) / 1024:.1f} KB)")
    return quant_path


def verify_quantized_model(
    float_path: str,
    quant_path: str,
    calibration_data: list[dict],
    model_type: str = "policy",
    num_samples: int = 1000,
):
    """
    Compare float32 vs INT8 quantized model outputs.
    Flags if mean value diff > 0.01 or policy KL divergence > 0.05.
    """
    import onnxruntime as ort

    float_sess = ort.InferenceSession(float_path)
    quant_sess = ort.InferenceSession(quant_path)

    value_diffs = []
    kl_divs = []

    for i, d in enumerate(calibration_data[:num_samples]):
        if model_type == "wp":
            t0 = heroes_to_multi_hot(d["team0_heroes"])
            t1 = heroes_to_multi_hot(d["team1_heroes"])
            m = map_to_one_hot(d["game_map"])
            t = tier_to_one_hot(d["skill_tier"])
            x = np.concatenate([t0, t1, m, t]).reshape(1, -1).astype(np.float32)
            feeds = {"input": x}
        else:
            t0 = heroes_to_multi_hot(d.get("team0_heroes", []))
            t1 = heroes_to_multi_hot(d.get("team1_heroes", []))
            bans = np.zeros(NUM_HEROES, dtype=np.float32)
            for h in d.get("team0_bans", []) + d.get("team1_bans", []):
                idx = HERO_TO_IDX.get(h)
                if idx is not None:
                    bans[idx] = 1.0
            m = map_to_one_hot(d["game_map"])
            t = tier_to_one_hot(d["skill_tier"])
            state = np.concatenate([t0, t1, bans, m, t, [1.0, 1.0]]).reshape(1, -1).astype(np.float32)
            mask = np.ones((1, NUM_HEROES), dtype=np.float32)
            taken = set(d.get("team0_heroes", []) + d.get("team1_heroes", []) +
                       d.get("team0_bans", []) + d.get("team1_bans", []))
            for h in taken:
                idx = HERO_TO_IDX.get(h)
                if idx is not None:
                    mask[0, idx] = 0.0
            feeds = {"state": state, "valid_mask": mask}

        float_out = float_sess.run(None, feeds)
        quant_out = quant_sess.run(None, feeds)

        if model_type == "wp":
            value_diffs.append(abs(float_out[0][0] - quant_out[0][0]))
        elif model_type == "policy":
            # Value head diff
            value_diffs.append(abs(float_out[1][0] - quant_out[1][0]))
            # Policy KL divergence
            f_logits = float_out[0][0]
            q_logits = quant_out[0][0]
            # Softmax
            f_probs = np.exp(f_logits - f_logits.max()) / np.exp(f_logits - f_logits.max()).sum()
            q_probs = np.exp(q_logits - q_logits.max()) / np.exp(q_logits - q_logits.max()).sum()
            # KL(float || quant)
            kl = np.sum(f_probs * np.log(np.clip(f_probs, 1e-10, 1) / np.clip(q_probs, 1e-10, 1)))
            kl_divs.append(kl)
        else:  # gd
            f_logits = float_out[0][0]
            q_logits = quant_out[0][0]
            f_probs = np.exp(f_logits - f_logits.max()) / np.exp(f_logits - f_logits.max()).sum()
            q_probs = np.exp(q_logits - q_logits.max()) / np.exp(q_logits - q_logits.max()).sum()
            kl = np.sum(f_probs * np.log(np.clip(f_probs, 1e-10, 1) / np.clip(q_probs, 1e-10, 1)))
            kl_divs.append(kl)

    mean_val_diff = np.mean(value_diffs) if value_diffs else 0
    max_val_diff = np.max(value_diffs) if value_diffs else 0
    mean_kl = np.mean(kl_divs) if kl_divs else 0
    max_kl = np.max(kl_divs) if kl_divs else 0

    print(f"  Quantization verification ({num_samples} samples):")
    if value_diffs:
        print(f"    Value: mean_diff={mean_val_diff:.6f} max_diff={max_val_diff:.6f}"
              f" {'⚠ EXCEEDS THRESHOLD' if mean_val_diff > 0.01 else '✓'}")
    if kl_divs:
        print(f"    Policy KL: mean={mean_kl:.6f} max={max_kl:.6f}"
              f" {'⚠ EXCEEDS THRESHOLD' if mean_kl > 0.05 else '✓'}")

    return {
        "mean_value_diff": mean_val_diff,
        "max_value_diff": max_val_diff,
        "mean_kl": mean_kl,
        "max_kl": max_kl,
    }
