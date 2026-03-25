"""
Win Probability model sanity tests.

Tests absurd compositions, high-WR heroes in bad comps, clearly better comps,
and symmetry. Any WP model should pass these — they represent game knowledge
that is unambiguous.

Usage:
    python training/test_wp_sanity.py                          # test current model
    python training/test_wp_sanity.py training/wp_sweep_42.pt  # test a sweep checkpoint
    python training/test_wp_sanity.py --embedding 42           # test embedding model from sweep
"""
import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from shared import heroes_to_multi_hot, map_to_one_hot, tier_to_one_hot, NUM_HEROES, HERO_TO_IDX


def load_model(path=None, embedding_config=None):
    """Load a WP model. Returns (model, is_embedding, eval_fn)."""
    from train_win_probability import WinProbModel, WinProbEmbeddingModel

    if embedding_config:
        model = WinProbEmbeddingModel(**embedding_config)
        model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        model.eval()

        def eval_fn(t0_heroes, t1_heroes, game_map='Cursed Hollow', tier='mid'):
            t0_idx = torch.tensor([[HERO_TO_IDX[h] for h in t0_heroes[:5]]], dtype=torch.long)
            t1_idx = torch.tensor([[HERO_TO_IDX[h] for h in t1_heroes[:5]]], dtype=torch.long)
            t0_mask = torch.ones(1, 5)
            t1_mask = torch.ones(1, 5)
            t0_mask[0, len(t0_heroes):] = 0
            t1_mask[0, len(t1_heroes):] = 0
            ctx = torch.tensor([list(map_to_one_hot(game_map)) + list(tier_to_one_hot(tier))],
                              dtype=torch.float32)
            with torch.no_grad():
                return model(t0_idx, t1_idx, t0_mask, t1_mask, ctx).item()

        return model, True, eval_fn
    else:
        model = WinProbModel()
        if path:
            model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        else:
            default = os.path.join(os.path.dirname(__file__), 'win_probability.pt')
            model.load_state_dict(torch.load(default, weights_only=True, map_location='cpu'))
        model.eval()

        def eval_fn(t0_heroes, t1_heroes, game_map='Cursed Hollow', tier='mid'):
            t0 = heroes_to_multi_hot(t0_heroes)
            t1 = heroes_to_multi_hot(t1_heroes)
            m = map_to_one_hot(game_map)
            t = tier_to_one_hot(tier)
            x = torch.tensor([list(t0) + list(t1) + list(m) + list(t)], dtype=torch.float32)
            with torch.no_grad():
                return model(x).item()

        return model, False, eval_fn


# ── Test definitions ──

TESTS = []


def wp_test(name, t0, t1, expected_winner, game_map='Cursed Hollow', category=''):
    TESTS.append({
        'name': name,
        't0': t0,
        't1': t1,
        'expected_winner': expected_winner,
        'game_map': game_map,
        'category': category,
    })


def symmetry_test(name, comp_a, comp_b, game_map='Cursed Hollow'):
    TESTS.append({
        'name': name,
        't0': comp_a,
        't1': comp_b,
        'expected_winner': 'SYMMETRY',
        'game_map': game_map,
        'category': 'symmetry',
    })


# ── Absurd compositions vs standard ──

standard = ['Muradin', 'Brightwing', 'Valla', 'Sonya', 'Jaina']

wp_test('5 tanks vs standard',
        ['Muradin', 'Johanna', 'Diablo', 'E.T.C.', "Mal'Ganis"], standard,
        'T1', category='absurd')

wp_test('5 healers vs standard',
        ['Brightwing', 'Malfurion', 'Rehgar', 'Uther', 'Anduin'], standard,
        'T1', category='absurd')

wp_test('5 ranged assassins vs standard',
        ['Valla', 'Jaina', 'Li-Ming', 'Falstad', 'Raynor'], standard,
        'T1', category='absurd')

wp_test('4 ranged + 1 melee assassin vs standard',
        ['Valla', 'Jaina', 'Li-Ming', 'Falstad', 'Zeratul'], standard,
        'T1', category='absurd')

wp_test('5 bruisers vs standard',
        ['Sonya', 'Dehaka', 'Imperius', 'Hogger', 'Artanis'], standard,
        'T1', category='absurd')

wp_test('3 tanks 2 healers (no damage) vs standard',
        ['Muradin', 'Johanna', 'Diablo', 'Brightwing', 'Malfurion'], standard,
        'T1', category='absurd')

wp_test('5 melee assassins vs standard',
        ['Zeratul', 'Illidan', 'Kerrigan', 'Malthael', 'Qhira'], standard,
        'T1', category='absurd')

wp_test('4 supports + 1 healer vs standard',
        ['Medivh', 'Zarya', 'Brightwing', 'Tassadar', 'Tyrande'], standard,
        'T1', category='absurd')


# ── High WR heroes in bad comps (model trap) ──

wp_test('Top WR heroes (no healer, 2 tanks) vs balanced',
        ['Muradin', 'Johanna', 'Valla', 'Falstad', 'Li-Ming'],
        ['Diablo', 'Brightwing', 'Raynor', 'Sonya', 'Jaina'],
        'T1', category='trap')

wp_test('All bruisers high WR vs balanced',
        ['Sonya', 'Dehaka', 'Hogger', 'Imperius', 'Yrel'],
        ['Muradin', 'Rehgar', 'Valla', 'Falstad', 'Sonya'],
        'T1', category='trap')

wp_test('Double healer wasted slot vs single healer',
        ['Muradin', 'Brightwing', 'Malfurion', 'Valla', 'Sonya'],
        ['Johanna', 'Rehgar', 'Li-Ming', 'Falstad', 'Dehaka'],
        'T1', category='trap')

wp_test('Triple tank no healer vs standard',
        ['Muradin', 'Johanna', "Mal'Ganis", 'Valla', 'Jaina'],
        ['Diablo', 'Brightwing', 'Li-Ming', 'Sonya', 'Falstad'],
        'T1', category='trap')

wp_test('No frontline (all ranged + healer) vs balanced',
        ['Brightwing', 'Valla', 'Jaina', 'Li-Ming', 'Falstad'],
        ['Muradin', 'Rehgar', 'Raynor', 'Sonya', 'Thrall'],
        'T1', category='trap')


# ── Clearly better normal comps ──

wp_test('Meta comp vs off-meta (Braxis)',
        ['Johanna', 'Brightwing', 'Falstad', 'Sonya', 'Jaina'],
        ['Murky', 'Probius', 'The Lost Vikings', 'Nova', 'Samuro'],
        'T0', game_map='Braxis Holdout', category='normal')

wp_test('Wombo combo vs random decent heroes',
        ['E.T.C.', 'Brightwing', 'Jaina', "Kael'thas", 'Sonya'],
        ['Muradin', 'Malfurion', 'Raynor', 'Artanis', 'Tychus'],
        'T0', category='normal')

wp_test('Macro comp on Garden vs teamfight comp',
        ['Dehaka', 'Brightwing', 'Falstad', 'Sonya', 'Zagara'],
        ['E.T.C.', 'Uther', 'Jaina', "Kael'thas", 'Valla'],
        'T0', game_map='Garden of Terror', category='normal')

wp_test('Triple melee vs poke/kite comp',
        ['Diablo', 'Sonya', 'Thrall', 'Rehgar', 'Illidan'],
        ['Johanna', 'Brightwing', 'Li-Ming', 'Falstad', 'Raynor'],
        'T1', category='normal')

wp_test('Strong balanced vs weak balanced',
        ['Johanna', 'Malfurion', 'Li-Ming', 'Dehaka', 'Valla'],
        ['Murky', 'Lt. Morales', 'Nova', 'Gazlowe', 'Chen'],
        'T0', category='normal')


# ── Symmetry tests ──

symmetry_test('Symmetry: balanced vs balanced',
              ['Muradin', 'Brightwing', 'Valla', 'Sonya', 'Jaina'],
              ['Johanna', 'Malfurion', 'Li-Ming', 'Dehaka', 'Falstad'])

symmetry_test('Symmetry: aggressive vs aggressive',
              ['Diablo', 'Rehgar', 'Valla', 'Thrall', 'Falstad'],
              ['Johanna', 'Brightwing', 'Li-Ming', 'Sonya', 'Raynor'])

symmetry_test('Symmetry: double healer vs double bruiser',
              ['Muradin', 'Brightwing', 'Malfurion', 'Valla', 'Falstad'],
              ['Johanna', 'Sonya', 'Dehaka', 'Li-Ming', 'Rehgar'])

# Additional mirror/symmetry tests: different comp strengths
symmetry_test('Symmetry: degenerate vs standard',
              ['Muradin', 'Johanna', 'Diablo', 'E.T.C.', "Mal'Ganis"],
              ['Muradin', 'Brightwing', 'Valla', 'Sonya', 'Jaina'])

symmetry_test('Symmetry: no-healer vs healer',
              ['Muradin', 'Johanna', 'Valla', 'Falstad', 'Li-Ming'],
              ['Diablo', 'Brightwing', 'Raynor', 'Sonya', 'Jaina'])

symmetry_test('Symmetry: off-meta vs off-meta',
              ['Murky', 'Probius', 'The Lost Vikings', 'Nova', 'Samuro'],
              ['Gazlowe', 'Chen', 'Lt. Morales', 'Rexxar', 'Tyrande'])

symmetry_test('Symmetry: strong vs weak',
              ['Johanna', 'Malfurion', 'Li-Ming', 'Dehaka', 'Valla'],
              ['Murky', 'Lt. Morales', 'Nova', 'Gazlowe', 'Chen'])

# Mirror matches on different maps (should all be 50/50)
symmetry_test('Symmetry: standard vs standard (Braxis)',
              ['Johanna', 'Brightwing', 'Valla', 'Sonya', 'Falstad'],
              ['Diablo', 'Rehgar', 'Li-Ming', 'Dehaka', 'Raynor'],
              game_map='Braxis Holdout')

symmetry_test('Symmetry: tank-heavy vs ranged-heavy (Infernal)',
              ['Muradin', 'Johanna', 'Brightwing', 'Valla', 'Sonya'],
              ['Diablo', 'Rehgar', 'Li-Ming', 'Falstad', 'Raynor'],
              game_map='Infernal Shrines')

# Cross-map symmetry: same matchup should swap cleanly regardless of map
symmetry_test('Symmetry: balanced vs balanced (Towers)',
              ['Muradin', 'Brightwing', 'Valla', 'Sonya', 'Jaina'],
              ['Johanna', 'Malfurion', 'Li-Ming', 'Dehaka', 'Falstad'],
              game_map='Towers of Doom')


# ── Runner ──

def run_tests(eval_fn, verbose=True):
    """Run all tests. Returns (passed, total, results_list)."""
    results = []
    categories = {}

    for t in TESTS:
        name = t['name']
        t0, t1 = t['t0'], t['t1']
        game_map = t['game_map']
        category = t['category']

        if t['expected_winner'] == 'SYMMETRY':
            wp_ab = eval_fn(t0, t1, game_map)
            wp_ba = eval_fn(t1, t0, game_map)
            sym_diff = abs((wp_ab + wp_ba) - 1.0)
            passed = sym_diff < 0.05

            if verbose:
                status = 'PASS' if passed else 'FAIL'
                print(f'{status} {name}')
                print(f'  A vs B: {wp_ab:.1%} | B vs A: {wp_ba:.1%} | '
                      f'Sum: {wp_ab+wp_ba:.4f} | Asymmetry: {sym_diff:.4f}')
                print()
        else:
            wp0 = eval_fn(t0, t1, game_map)
            expected = t['expected_winner']
            if expected == 'T0':
                passed = wp0 > 0.5
            else:
                passed = wp0 < 0.5

            if verbose:
                status = 'PASS' if passed else 'FAIL'
                print(f'{status} {name}')
                print(f'  T0: {t0}')
                print(f'  T1: {t1}')
                margin = abs(wp0 - 0.5) * 100
                winner = 'T0' if wp0 > 0.5 else 'T1'
                print(f'  WP: {wp0:.1%} T0 | {1-wp0:.1%} T1 '
                      f'(expected {expected}, got {winner}, margin {margin:.1f}pp)')
                print()

        results.append(passed)
        categories.setdefault(category, []).append(passed)

    passed = sum(results)
    total = len(results)

    if verbose:
        print('=' * 60)
        print(f'TOTAL: {passed}/{total} passed')
        for cat in ['absurd', 'trap', 'normal', 'symmetry']:
            if cat in categories:
                cp = sum(categories[cat])
                ct = len(categories[cat])
                print(f'  {cat}: {cp}/{ct}')
        print('=' * 60)

    return passed, total, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs='?', default=None, help='Path to .pt checkpoint')
    parser.add_argument('--embedding', type=int, default=None,
                        help='Sweep variant ID (loads config from CSV)')
    args = parser.parse_args()

    if args.embedding is not None:
        # Load config from sweep CSV
        import csv, json
        csv_path = os.path.join(os.path.dirname(__file__), 'win_prob_sweep_results.csv')
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if int(row['variant_id']) == args.embedding:
                    config = {
                        'embed_dim': int(row['embed_dim']),
                        'interaction_mode': row['interaction_mode'],
                        'hidden_dims': json.loads(row['hidden_dims']),
                        'dropout': float(row['dropout']),
                    }
                    pt_path = os.path.join(os.path.dirname(__file__),
                                          f'wp_sweep_{args.embedding}.pt')
                    print(f"Loading embedding model v{args.embedding}: {row['name']}")
                    print(f"  Config: {config}")
                    print(f"  Reported acc: {row['best_test_acc']}%")
                    print()
                    _, _, eval_fn = load_model(pt_path, embedding_config=config)
                    run_tests(eval_fn)
                    sys.exit(0)
        print(f"Variant {args.embedding} not found in CSV")
        sys.exit(1)

    elif args.model_path:
        print(f"Loading model from {args.model_path}")
        _, _, eval_fn = load_model(args.model_path)
    else:
        print("Loading current WP model")
        _, _, eval_fn = load_model()

    print()
    run_tests(eval_fn)
