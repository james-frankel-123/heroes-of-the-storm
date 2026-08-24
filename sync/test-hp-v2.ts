/**
 * Fixture-mode test for the v1 client adapters. Runs against the new API
 * BEFORE live-data activation (responses are example data, cost no quota).
 * Asserts the legacy shapes every worker consumes.
 *
 *   set -a && source .env && set +a && npx tsx sync/test-hp-v2.ts
 */
import { HeroesProfileApiV2 } from './api-client-v2'

let pass = 0
let fail = 0
function ok(name: string, cond: boolean, detail = '') {
  if (cond) { pass++; console.log(`  ok   ${name}`) }
  else { fail++; console.log(`  FAIL ${name} ${detail}`) }
}

async function main() {
  const key = process.env.HEROES_PROFILE_V2_API_KEY
  if (!key) { console.error('HEROES_PROFILE_V2_API_KEY required'); process.exit(1) }
  const api = new HeroesProfileApiV2(key, 60, 2)

  console.log('replay detail (legacy Replay/Data shape):')
  const replay = await api.getReplayData(90000001)
  ok('fixture mode', api.lastDataSource === 'fixture', `got ${api.lastDataSource}`)
  const playerEntries = Object.entries(replay).filter(([, v]: any) =>
    v && typeof v === 'object' && 'blizz_id' in v && 'hero' in v)
  ok('10 battletag-keyed player entries', playerEntries.length === 10, `got ${playerEntries.length}`)
  const [tag, p]: any = playerEntries[0]
  ok('battletag key has discriminator', tag.includes('#'), tag)
  ok('hero is a string name', typeof p.hero === 'string', String(p.hero))
  ok('team is 0/1', p.team === 0 || p.team === 1)
  ok('winner present', p.winner === 0 || p.winner === 1 || typeof p.winner === 'boolean')
  ok('mmr fields', typeof p.player_mmr === 'number')
  ok('scores object (legacy name)', p.scores && typeof p.scores === 'object' && 'takedowns' in p.scores)
  ok('talents object', p.talents && typeof p.talents === 'object')
  ok('game_map flattened to name', typeof replay.game_map === 'string')
  ok('draft_order heroes are strings', typeof replay.draft_order?.[0]?.hero === 'string')
  ok('draft_order has 16 steps', (replay.draft_order ?? []).length === 16, String(replay.draft_order?.length))
  ok('game_length is seconds number', typeof replay.game_length === 'number')

  console.log('replays listing (legacy Min_id contract):')
  const rows = await api.getReplayMinId(90000000, 'Storm League', 50)
  ok('rows returned', rows.length > 0, String(rows.length))
  ok('rows have replayID', typeof rows[0]?.replayID === 'number')
  ok('valid synthesized', rows[0]?.valid === 1 || rows[0]?.valid === 0)
  ok('rows have game_version', 'game_version' in (rows[0] ?? {}))
  ok('ids ascend from inclusive min', rows.every((r: any) => r.replayID >= 90000000))

  console.log('replay max:')
  const max = await api.getReplayMax()
  ok('max id is a number', Number.isFinite(max) && max > 0, String(max))

  console.log('player matches (legacy Player/Replays contract):')
  const pr = await api.getPlayerReplays('Zemill#1940', 1, undefined, undefined, 'Storm League')
  const inner = pr['Storm League'] ?? {}
  ok('game-type keyed object', typeof inner === 'object')
  ok('replay ids as keys', Object.keys(inner).length > 0 && Object.keys(inner).every(k => /^\d+$/.test(k)),
     `${Object.keys(inner).length} keys`)

  console.log('hero stats (202-tolerant):')
  const hs = await api.getHeroStats('minor', '2.55.17.97771')
  ok('rows array', Array.isArray(hs) && hs.length > 0, `type ${typeof hs}`)

  console.log('hero matchups:')
  const mu: any = await api.getHeroMatchups('Jaina', 'minor', '2.55.17.97771')
  ok('ally/enemy sections', Array.isArray(mu.ally) && Array.isArray(mu.enemy))

  console.log(`\n${pass} passed, ${fail} failed`)
  process.exit(fail === 0 ? 0 : 1)
}
main()
