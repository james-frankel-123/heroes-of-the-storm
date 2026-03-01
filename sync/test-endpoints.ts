import { HeroesProfileApi } from './api-client'

const api = new HeroesProfileApi(process.env.HEROES_PROFILE_API_KEY!)

interface TestResult {
  endpoint: string
  path: string
  status: 'OK' | 'FAIL'
  elapsed: string
  detail: string
}

async function testEndpoint(
  name: string,
  path: string,
  fn: () => Promise<any>,
): Promise<TestResult> {
  const start = Date.now()
  try {
    const data = await fn()
    const elapsed = ((Date.now() - start) / 1000).toFixed(1) + 's'
    let detail: string
    if (Array.isArray(data)) {
      detail = `Array[${data.length}]`
    } else if (typeof data === 'object' && data !== null) {
      const keys = Object.keys(data)
      detail = `Object{${keys.length} keys} (${keys.slice(0, 3).join(', ')}${keys.length > 3 ? '...' : ''})`
    } else {
      detail = String(data).slice(0, 100)
    }
    return { endpoint: name, path, status: 'OK', elapsed, detail }
  } catch (err) {
    const elapsed = ((Date.now() - start) / 1000).toFixed(1) + 's'
    const msg = err instanceof Error ? err.message : String(err)
    return { endpoint: name, path, status: 'FAIL', elapsed, detail: msg }
  }
}

async function main() {
  const results: TestResult[] = []

  // 1. Patches
  results.push(await testEndpoint(
    'Patches',
    'Patches',
    () => api.getPatches(),
  ))

  // 2. Hero Stats (one tier)
  results.push(await testEndpoint(
    'Hero Stats (low)',
    'Heroes/Stats?league_tier=1,2',
    () => api.getHeroStats('major', '2.55', '1,2'),
  ))

  // 3. Talent Details (one tier)
  results.push(await testEndpoint(
    'Talent Details (low)',
    'Heroes/Talents/Details?league_tier=1,2',
    () => api.getTalentDetails('major', '2.55', '1,2'),
  ))

  // 4. Hero Matchups (one hero)
  results.push(await testEndpoint(
    'Matchups (Ana)',
    'Heroes/Matchups?hero=Ana',
    () => api.getHeroMatchups('Ana', 'major', '2.55'),
  ))

  // 5. Player Replays
  results.push(await testEndpoint(
    'Player Replays (Django)',
    'Player/Replays?battletag=Django#1458',
    () => api.getPlayerReplays('Django#1458', 1),
  ))

  // 6. Player Hero All
  results.push(await testEndpoint(
    'Player Hero All (Django)',
    'Player/Hero/All?battletag=Django#1458',
    () => api.getPlayerHeroAll('Django#1458', 1),
  ))

  // Print results table
  console.log('\n' + '='.repeat(120))
  console.log('ENDPOINT TEST RESULTS')
  console.log('='.repeat(120))
  for (const r of results) {
    const statusIcon = r.status === 'OK' ? 'PASS' : 'FAIL'
    console.log(`[${statusIcon}] ${r.endpoint.padEnd(30)} ${r.elapsed.padStart(8)}  ${r.detail.slice(0, 200)}`)
  }
  console.log('='.repeat(120))
  console.log(`Total API calls: ${api.getCallCount()}`)

  const failures = results.filter(r => r.status === 'FAIL')
  if (failures.length > 0) {
    console.log(`\n${failures.length} FAILURES:`)
    for (const f of failures) {
      console.log(`\n--- ${f.endpoint} (${f.path}) ---`)
      console.log(f.detail)
    }
  }
}

main().catch(e => { console.error(e); process.exit(1) })
