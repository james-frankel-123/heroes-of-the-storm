/**
 * Account-level refusal from Heroes Profile (401/403): bad key, plan
 * doesn't cover the endpoint, or — since 2026-10-01 — terms_not_accepted /
 * project_details_required. Nothing is wrong with the replay/player being
 * requested, so workers must NOT mark items as permanently skipped or move
 * cursors past them; they stop and let the next cron run retry.
 */
export class HpAccessPausedError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    detail: string,
  ) {
    super(`Heroes Profile access paused (${status} ${code}): ${detail}`)
    this.name = 'HpAccessPausedError'
  }
}

export function isHpAccessPaused(err: unknown): err is HpAccessPausedError {
  return err instanceof HpAccessPausedError ||
    (err instanceof Error && err.name === 'HpAccessPausedError')
}
