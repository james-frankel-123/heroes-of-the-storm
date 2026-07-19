// Parity oracle (MCTS behavioral layer). Runs the REAL TypeScript MCTS
// (src/lib/draft/mcts-search.ts) through onnxruntime-web, driven by a shared
// mock RNG + fixed sim count (uncapped time). The C# engine runs the same
// states through the same RNG sequence and is compared for behavioral agreement.
//
// Because the C# and web model outputs differ by ~1e-6 (see the model-layer
// oracle), discrete branch points (UCB argmax, GD inverse-CDF threshold) can
// occasionally desync the shared RNG stream — so agreement is near-exact but
// statistical, not bit-identical. Run: npm run gen:mcts

import * as ort from 'onnxruntime-web';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import { runMCTSSearch } from '../../src/lib/draft/mcts-search';

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODELS = join(__dirname, '..', '..', 'public', 'models');
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = pathToFileURL(join(__dirname, 'node_modules', 'onnxruntime-web', 'dist')).href + '/';

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
const DRAFT_ORDER: [number, 'ban' | 'pick'][] = [
  [0,'ban'],[1,'ban'],[0,'ban'],[1,'ban'],
  [0,'pick'],[1,'pick'],[1,'pick'],[0,'pick'],[0,'pick'],
  [1,'ban'],[0,'ban'],
  [1,'pick'],[1,'pick'],[0,'pick'],[0,'pick'],[1,'pick'],
];

function mulberry32(a: number) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

const MAX_SIMS = 60, MIN_SIMS = 5;

// Shared mock-RNG sequence (both engines step through this, cycling).
const seqGen = mulberry32(0xC0FFEE);
const rngSequence = Array.from({ length: 512 }, () => seqGen());

// Build cases: states sitting at OUR pick decision, with heroes already drafted.
const caseGen = mulberry32(0x1234ABCD);
const pick = <T>(arr: T[]) => arr[Math.floor(caseGen() * arr.length)];
const NUM_CASES = 20;
const metas: any[] = [];
for (let c = 0; c < NUM_CASES; c++) {
  const ourTeam = caseGen() < 0.5 ? 0 : 1;
  // choose a step that is our team's pick
  const ourPickSteps = DRAFT_ORDER.map((d, i) => [i, d] as const).filter(([, d]) => d[0] === ourTeam && d[1] === 'pick').map(([i]) => i);
  const step = ourPickSteps[Math.floor(caseGen() * ourPickSteps.length)];
  const map = pick(MAPS), tier = pick(TIERS);
  const used = new Set<string>(); const t0: string[] = [], t1: string[] = [], bans: string[] = [];
  for (let s = 0; s < step; s++) {
    let hero: string; do { hero = pick(HEROES); } while (used.has(hero));
    used.add(hero);
    const [team, type] = DRAFT_ORDER[s];
    if (type === 'ban') bans.push(hero); else if (team === 0) t0.push(hero); else t1.push(hero);
  }
  metas.push({ team0Picks: t0, team1Picks: t1, bans, map, tier, step, ourTeam });
}

// Load models via onnxruntime-web.
const load = async (f: string) => ort.InferenceSession.create(new Uint8Array(readFileSync(join(MODELS, f))), { executionProviders: ['wasm'] });
const policy = await load('draft_policy.onnx');
const gd = await load('generic_draft_0.onnx');
console.log(`Running ${NUM_CASES} MCTS cases (${MAX_SIMS} sims each) through the real TS engine...`);

const round6 = (x: number) => Math.round(x * 1e6) / 1e6;
const cases: any[] = [];
for (const m of metas) {
  let idx = 0;
  const rng = () => rngSequence[idx++ % rngSequence.length];
  const taken = new Set<string>([...m.team0Picks, ...m.team1Picks, ...m.bans]);
  const res = await runMCTSSearch(
    ort, policy, gd,
    { team0Picks: m.team0Picks, team1Picks: m.team1Picks, bans: m.bans, map: m.map, tier: m.tier, step: m.step, ourTeam: m.ourTeam, stepType: 'pick' },
    taken, undefined, undefined,
    { rng, maxSims: MAX_SIMS, minSims: MIN_SIMS, timeBudgetMs: Infinity },
  );
  cases.push({
    meta: m,
    valueEstimate: round6(res.valueEstimate),
    sims: res.sims,
    recommendations: res.recommendations.map(r => ({ hero: r.hero, visits: round6(r.visits), q: round6(r.q) })),
  });
}

const out = { _meta: { generator: 'onnxruntime-web + real TS runMCTSSearch', maxSims: MAX_SIMS, minSims: MIN_SIMS, numCases: NUM_CASES }, rngSequence, cases };
const outPath = join(__dirname, 'mcts-golden.json');
writeFileSync(outPath, JSON.stringify(out));
console.log(`Wrote ${NUM_CASES} cases -> ${outPath}`);
