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
