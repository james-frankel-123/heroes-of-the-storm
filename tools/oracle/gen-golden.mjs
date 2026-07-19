// Parity oracle — generates model I/O golden vectors from onnxruntime-web (the
// runtime production actually uses in the browser). The C# engine is validated
// within-margin against these outputs (Layer 2), and re-encodes each case's
// inputs to confirm byte-identical tensors (Layer 1).
//
// Deterministic: a seeded PRNG drives case generation and model weights are
// fixed, so regenerating produces an identical file. Run: npm run gen

import * as ort from 'onnxruntime-web';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODELS = join(__dirname, '..', '..', 'public', 'models');

// Single-threaded wasm for reproducibility; point at the installed wasm assets.
// On Windows the loader needs a file:// URL prefix, not a bare drive path.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = pathToFileURL(join(__dirname, 'node_modules', 'onnxruntime-web', 'dist')).href + '/';

// ── Encoding vocab (must match ai-inference.ts / training) ───────────
const HEROES = [
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
];
const MAPS = [
  "Alterac Pass","Battlefield of Eternity","Blackheart's Bay","Braxis Holdout",
  "Cursed Hollow","Dragon Shire","Garden of Terror","Hanamura Temple",
  "Infernal Shrines","Sky Temple","Tomb of the Spider Queen","Towers of Doom",
  "Volskaya Foundry","Warhead Junction",
];
const TIERS = ["low", "mid", "high"];
const H = HEROES.length, M = MAPS.length, T = TIERS.length;
const HIDX = Object.fromEntries(HEROES.map((h, i) => [h, i]));

// (team, type) for the 16 draft steps — matches DRAFT_ORDER in mcts-search.ts
const DRAFT_ORDER = [
  [0,'ban'],[1,'ban'],[0,'ban'],[1,'ban'],
  [0,'pick'],[1,'pick'],[1,'pick'],[0,'pick'],[0,'pick'],
  [1,'ban'],[0,'ban'],
  [1,'pick'],[1,'pick'],[0,'pick'],[0,'pick'],[1,'pick'],
];

// ── Encoders (mirror ai-inference.ts exactly) ────────────────────────
function multiHot(names, out, off) { for (const n of names) { const i = HIDX[n]; if (i !== undefined) out[off + i] = 1; } }
function encodeState(t0, t1, bans, map, tier, step, isPick, ourTeam) {
  const v = new Float32Array(290); let o = 0;
  multiHot(t0, v, o); o += H; multiHot(t1, v, o); o += H; multiHot(bans, v, o); o += H;
  const mi = MAPS.indexOf(map); if (mi >= 0) v[o + mi] = 1; o += M;
  const ti = TIERS.indexOf(tier); if (ti >= 0) v[o + ti] = 1; o += T;
  v[o++] = step / 15.0; v[o++] = isPick ? 1 : 0; v[o++] = ourTeam;
  return v;
}
function validMask(taken) { const m = new Float32Array(H).fill(1); for (const n of taken) { const i = HIDX[n]; if (i !== undefined) m[i] = 0; } return m; }
function wpBase(t0, t1, map, tier) {
  const v = new Float32Array(197); let o = 0;
  multiHot(t0, v, o); o += H; multiHot(t1, v, o); o += H;
  const mi = MAPS.indexOf(map); if (mi >= 0) v[o + mi] = 1; o += M;
  const ti = TIERS.indexOf(tier); if (ti >= 0) v[o + ti] = 1;
  return v;
}

// ── Seeded PRNG (mulberry32) ─────────────────────────────────────────
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rand = mulberry32(0x9E3779B9);
const pick = (arr) => arr[Math.floor(rand() * arr.length)];
const round6 = (x) => Math.round(x * 1e6) / 1e6;

// ── Build cases ──────────────────────────────────────────────────────
const N = 100;
const cases = [];
for (let c = 0; c < N; c++) {
  const step = Math.floor(rand() * 16);
  const map = pick(MAPS), tier = pick(TIERS), ourTeam = rand() < 0.5 ? 0 : 1;

  // Fill the draft up to `step` with distinct random heroes, honoring DRAFT_ORDER.
  const used = new Set(); const t0 = [], t1 = [], bans = [];
  for (let s = 0; s < step; s++) {
    let hero; do { hero = pick(HEROES); } while (used.has(hero));
    used.add(hero);
    const [team, type] = DRAFT_ORDER[s];
    if (type === 'ban') bans.push(hero); else if (team === 0) t0.push(hero); else t1.push(hero);
  }
  const isPick = DRAFT_ORDER[step][1] === 'pick';
  const taken = [...t0, ...t1, ...bans];

  const policyState = encodeState(t0, t1, bans, map, tier, step, isPick, ourTeam);
  const mask = validMask(taken);
  const gdState = policyState.slice(0, 289);

  // WP input: base(197) + deterministic enriched tail (86) to exercise the full input surface.
  const wpInput = new Float32Array(283);
  wpInput.set(wpBase(t0, t1, map, tier), 0);
  for (let i = 197; i < 283; i++) wpInput[i] = (rand() * 6 - 3); // pattern in [-3,3]

  cases.push({
    meta: { team0Picks: t0, team1Picks: t1, bans, map, tier, step, isPick, ourTeam },
    policyState: Array.from(policyState),
    mask: Array.from(mask),
    gdState: Array.from(gdState),
    wpInput: Array.from(wpInput).map(round6),
  });
}

// ── Run the three models via onnxruntime-web ─────────────────────────
console.log(`onnxruntime-web ${ort.env.versions.web ?? ort.version ?? ''} — running ${N} cases...`);
const load = async (f) => ort.InferenceSession.create(new Uint8Array(readFileSync(join(MODELS, f))), { executionProviders: ['wasm'] });
const policy = await load('draft_policy.onnx');
const gd = await load('generic_draft_0.onnx');
const wp = await load('win_probability.onnx');

for (const cs of cases) {
  const pr = await policy.run({
    state: new ort.Tensor('float32', Float32Array.from(cs.policyState), [1, 290]),
    valid_mask: new ort.Tensor('float32', Float32Array.from(cs.mask), [1, H]),
  });
  const gr = await gd.run({
    state: new ort.Tensor('float32', Float32Array.from(cs.gdState), [1, 289]),
    valid_mask: new ort.Tensor('float32', Float32Array.from(cs.mask), [1, H]),
  });
  const wr = await wp.run({ input: new ort.Tensor('float32', Float32Array.from(cs.wpInput), [1, 283]) });

  cs.outputs = {
    policyLogits: Array.from(pr.policy_logits.data).map(round6),
    value: round6(pr.value.data[0]),
    heroLogits: Array.from(gr.hero_logits.data).map(round6),
    winProbability: round6(wr.win_probability.data[0]),
  };
}

const out = {
  _meta: {
    generator: 'onnxruntime-web',
    ortVersion: '1.20.1',
    numCases: N,
    note: 'Reference model I/O from the production web runtime. Regenerate with `npm run gen` after a model change.',
  },
  cases,
};
const outPath = join(__dirname, 'oracle-golden.json');
writeFileSync(outPath, JSON.stringify(out));
console.log(`Wrote ${N} cases -> ${outPath} (${(JSON.stringify(out).length / 1e6).toFixed(2)} MB)`);
