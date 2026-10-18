/**
 * Neon cache for player lookups (personal_profiles, personal_metric_evidence).
 * Server only. Bounded: at most MAX_PROFILES rows; the oldest summaries go first.
 */
import { sql } from 'drizzle-orm'
import { db } from '@/lib/db'
import type { PlayerSummary } from './hp'
import type { MetricEvidence, Profile } from './model'

export const MAX_PROFILES = Number(process.env.PERSONAL_MAX_PROFILES ?? 2000)
export const SUMMARY_TTL_H = Number(process.env.PERSONAL_SUMMARY_TTL_H ?? 24)
export const PROFILE_TTL_H = Number(process.env.PERSONAL_PROFILE_TTL_H ?? 24)

export interface CachedProfile {
  region: number
  blizzId: number
  battletag: string
  summary: PlayerSummary
  summaryAt: string
  profile: Profile | null
  profileAt: string | null
  historySource: string | null
  wantsMetric: boolean
}

function rowsOf<T>(res: unknown): T[] {
  if (Array.isArray(res)) return res as T[]
  return ((res as { rows?: T[] })?.rows ?? []) as T[]
}

function toCached(r: any): CachedProfile {
  return {
    region: Number(r.region),
    blizzId: Number(r.blizz_id),
    battletag: r.battletag,
    summary: r.summary,
    summaryAt: new Date(r.summary_at).toISOString(),
    profile: r.profile ?? null,
    profileAt: r.profile_at ? new Date(r.profile_at).toISOString() : null,
    historySource: r.history_source ?? null,
    wantsMetric: !!r.wants_metric,
  }
}

export function nameOf(battletag: string): string {
  return battletag.split('#')[0].trim().toLowerCase().slice(0, 32)
}

export function isFresh(at: string | null, ttlHours: number): boolean {
  return !!at && Date.now() - Date.parse(at) < ttlHours * 3_600_000
}

/** Cached accounts whose name (the part before #) matches, case-insensitive. */
export async function findByName(name: string, limit = 20): Promise<CachedProfile[]> {
  const res = await db.execute(sql`
    SELECT region, blizz_id, battletag, summary, summary_at, NULL::jsonb AS profile, NULL AS profile_at,
           history_source, wants_metric
    FROM personal_profiles WHERE name_lower = ${name.toLowerCase().slice(0, 32)}
    ORDER BY (summary->>'games')::int DESC NULLS LAST LIMIT ${limit}`)
  return rowsOf<any>(res).map(toCached)
}

export async function findByTag(battletag: string, region: number): Promise<CachedProfile | null> {
  const res = await db.execute(sql`
    SELECT * FROM personal_profiles WHERE battletag = ${battletag} AND region = ${region} LIMIT 1`)
  const r = rowsOf<any>(res)[0]
  return r ? toCached(r) : null
}

export async function saveSummary(s: PlayerSummary): Promise<void> {
  await db.execute(sql`
    INSERT INTO personal_profiles (region, blizz_id, battletag, name_lower, summary, summary_at)
    VALUES (${s.region}, ${s.blizzId}, ${s.battletag}, ${nameOf(s.battletag)}, ${JSON.stringify(s)}::jsonb, now())
    ON CONFLICT (region, blizz_id) DO UPDATE SET
      battletag = excluded.battletag, name_lower = excluded.name_lower,
      summary = excluded.summary, summary_at = excluded.summary_at`)
  await evict()
}

/** Profile JSON with skill values rounded to 1e-5 (0.001 percentage points): about half the bytes. */
export function compactJson(p: Profile): string {
  return JSON.stringify(p, (_k, v) => (typeof v === 'number' && !Number.isInteger(v) ? Math.round(v * 1e5) / 1e5 : v))
}

export async function saveProfile(region: number, blizzId: number, p: Profile, source: string, wantsMetric: boolean): Promise<void> {
  await db.execute(sql`
    UPDATE personal_profiles SET profile = ${compactJson(p)}::jsonb, profile_at = now(),
      history_source = ${source}, wants_metric = ${wantsMetric}
    WHERE region = ${region} AND blizz_id = ${blizzId}`)
}

/** Heroes Profile says the account is private (or gone): drop everything we hold for it. */
export async function forget(battletag: string, region: number): Promise<void> {
  await db.execute(sql`
    DELETE FROM personal_metric_evidence e USING personal_profiles p
    WHERE p.battletag = ${battletag} AND p.region = ${region}
      AND e.region = p.region AND e.blizz_id = p.blizz_id`)
  await db.execute(sql`DELETE FROM personal_profiles WHERE battletag = ${battletag} AND region = ${region}`)
}

async function evict(): Promise<void> {
  await db.execute(sql`
    WITH old AS (
      SELECT region, blizz_id FROM personal_profiles
      ORDER BY summary_at DESC OFFSET ${MAX_PROFILES}
    ), gone AS (
      DELETE FROM personal_profiles p USING old
      WHERE p.region = old.region AND p.blizz_id = old.blizz_id
      RETURNING p.region, p.blizz_id
    )
    DELETE FROM personal_metric_evidence e USING gone
    WHERE e.region = gone.region AND e.blizz_id = gone.blizz_id`)
}

export async function metricEvidence(region: number, blizzId: number, modelVersion: string): Promise<MetricEvidence | null> {
  const res = await db.execute(sql`
    SELECT lam, eta FROM personal_metric_evidence
    WHERE region = ${region} AND blizz_id = ${blizzId} AND model_version = ${modelVersion}`)
  const r = rowsOf<{ lam: number[]; eta: number[] }>(res)[0]
  if (!r || r.lam.length !== 6 || r.eta.length !== 3) return null
  const [a, b, c, d, e, f] = r.lam.map(Number)
  return { Lam: [[a, b, c], [b, d, e], [c, e, f]], eta: r.eta.map(Number) }
}
