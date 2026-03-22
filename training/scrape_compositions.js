/**
 * Scrape composition data from Heroes Profile using headless Chromium.
 * The internal API endpoint has server-side bugs, so we intercept the XHR
 * response that the actual page makes.
 *
 * Usage:
 *   node training/scrape_compositions.js
 *
 * Output: src/lib/data/compositions-qm.json and updates compositions.json
 */
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const HP_BASE = 'https://www.heroesprofile.com';

// Tier configs to scrape
const CONFIGS = [
  // SL by tier group (matching our low/mid/high)
  { label: 'sl_low', game_type: 'sl', tiers: 'Bronze,Silver,Wood', min_games: 50 },
  { label: 'sl_mid', game_type: 'sl', tiers: 'Gold,Platinum', min_games: 100 },
  { label: 'sl_high', game_type: 'sl', tiers: 'Diamond,Master', min_games: 50 },
  // QM by tier group
  { label: 'qm_low', game_type: 'qm', tiers: 'Bronze,Silver,Wood', min_games: 50 },
  { label: 'qm_mid', game_type: 'qm', tiers: 'Gold,Platinum', min_games: 50 },
  { label: 'qm_high', game_type: 'qm', tiers: 'Diamond,Master', min_games: 50 },
];

function normalize(raw) {
  const roles = [
    raw.role_one?.name, raw.role_two?.name, raw.role_three?.name,
    raw.role_four?.name, raw.role_five?.name,
  ].filter(Boolean).sort();
  return {
    roles,
    winRate: Math.round(raw.win_rate * 100) / 100,
    games: raw.games_played,
    popularity: Math.round(raw.popularity * 100) / 100,
  };
}

async function scrapeConfig(browser, config) {
  const page = await browser.newPage();
  page.setDefaultTimeout(120000);

  // Intercept the compositions API response
  let compositionData = null;
  page.on('response', async (response) => {
    const url = response.url();
    if (url.includes('/api/v1/global/compositions') && response.request().method() === 'POST') {
      try {
        const data = await response.json();
        if (Array.isArray(data)) {
          compositionData = data;
          console.log(`  [${config.label}] Intercepted ${data.length} compositions`);
        }
      } catch (e) {
        console.log(`  [${config.label}] Response parse error:`, e.message);
      }
    }
  });

  const url = `${HP_BASE}/Global/Compositions?timeframe_type=major&timeframe=2.55&game_type=${config.game_type}&league_tier=${config.tiers}&minimum_games=${config.min_games}&mirror=0`;
  console.log(`[${config.label}] Loading: ${url}`);

  try {
    await page.goto(url, { waitUntil: 'networkidle2', timeout: 120000 });

    // Wait for the composition data to load (the page makes an XHR after loading)
    // Give it extra time since it can be slow
    let waited = 0;
    while (!compositionData && waited < 60000) {
      await new Promise(r => setTimeout(r, 2000));
      waited += 2000;
    }

    if (!compositionData) {
      console.log(`  [${config.label}] No data received after ${waited}ms`);
      await page.close();
      return [];
    }

    const comps = compositionData.map(normalize);
    const withHealer = comps.filter(c => c.roles.includes('Healer')).length;
    const noHealer = comps.filter(c => !c.roles.includes('Healer')).length;
    console.log(`  [${config.label}] ${comps.length} compositions (${withHealer} w/healer, ${noHealer} w/o)`);

    await page.close();
    return comps;
  } catch (e) {
    console.log(`  [${config.label}] Error: ${e.message}`);
    await page.close();
    return [];
  }
}

async function main() {
  console.log('Launching headless Chrome...');
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  const slResult = { low: [], mid: [], high: [] };
  const qmResult = { low: [], mid: [], high: [] };

  for (const config of CONFIGS) {
    const comps = await scrapeConfig(browser, config);
    const tier = config.label.split('_')[1]; // low/mid/high
    if (config.game_type === 'sl') {
      slResult[tier] = comps;
    } else {
      qmResult[tier] = comps;
    }
  }

  await browser.close();

  // Save SL compositions (update existing file)
  const slPath = path.join(__dirname, '../src/lib/data/compositions.json');
  fs.writeFileSync(slPath, JSON.stringify(slResult, null, 2) + '\n');
  console.log(`\nSaved SL compositions to ${slPath}`);
  for (const [tier, comps] of Object.entries(slResult)) {
    console.log(`  ${tier}: ${comps.length} compositions`);
  }

  // Save QM compositions
  const qmPath = path.join(__dirname, '../src/lib/data/compositions-qm.json');
  fs.writeFileSync(qmPath, JSON.stringify(qmResult, null, 2) + '\n');
  console.log(`Saved QM compositions to ${qmPath}`);
  for (const [tier, comps] of Object.entries(qmResult)) {
    console.log(`  ${tier}: ${comps.length} compositions`);
  }

  // Print summary
  console.log('\n=== SUMMARY ===');
  for (const [label, result] of [['SL', slResult], ['QM', qmResult]]) {
    for (const [tier, comps] of Object.entries(result)) {
      if (comps.length === 0) continue;
      const withH = comps.filter(c => c.roles.includes('Healer')).length;
      const noH = comps.filter(c => !c.roles.includes('Healer')).length;
      console.log(`${label} ${tier}: ${comps.length} comps (${withH} healer, ${noH} no-healer)`);
      // Top 3 by games
      const sorted = [...comps].sort((a, b) => b.games - a.games);
      for (const c of sorted.slice(0, 3)) {
        console.log(`  ${JSON.stringify(c.roles).padEnd(60)} wr=${c.winRate.toFixed(1)}% games=${c.games}`);
      }
      // Top no-healer comp
      const topNoHeal = sorted.filter(c => !c.roles.includes('Healer'));
      if (topNoHeal.length > 0) {
        console.log(`  Best no-healer: ${JSON.stringify(topNoHeal[0].roles)} wr=${topNoHeal[0].winRate.toFixed(1)}% games=${topNoHeal[0].games}`);
      }
    }
  }
}

main().catch(console.error);
