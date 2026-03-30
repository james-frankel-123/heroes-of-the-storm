/**
 * Main-thread coordinator for expectimax web workers.
 *
 * Splits root-level candidates across N workers, collects results,
 * and reports iterative deepening progress via callback.
 */

import type { DraftState, DraftData } from './types'
import { createSearchState, getValidHeroes, isBanPhase, isOurTurn } from './expectimax/search-state'
import { prefilterPicks, prefilterBans } from './expectimax/prefilter'
import type { ExpectimaxConfig, ExpectimaxResult, SearchState } from './expectimax/types'
import { DEFAULT_CONFIG } from './expectimax/types'

export class ExpectimaxManager {
  private workers: Worker[] = []
  private ready = false
  private searchId = 0

  /** Callback for iterative deepening progress */
  onProgress: ((results: ExpectimaxResult[], depth: number) => void) | null = null

  /** Number of workers to spawn */
  private numWorkers: number

  constructor(numWorkers?: number) {
    this.numWorkers = numWorkers ?? Math.min(
      typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 4 : 4,
      8,
    )
  }

  async init(): Promise<void> {
    if (this.ready) return

    const workerPromises: Promise<void>[] = []

    for (let i = 0; i < this.numWorkers; i++) {
      const worker = new Worker(
        new URL('./expectimax-worker.ts', import.meta.url),
        { type: 'module' },
      )

      const readyPromise = new Promise<void>((resolve, reject) => {
        const handler = (e: MessageEvent) => {
          if (e.data.type === 'ready') {
            worker.removeEventListener('message', handler)
            resolve()
          } else if (e.data.type === 'error') {
            worker.removeEventListener('message', handler)
            reject(new Error(e.data.message))
          }
        }
        worker.addEventListener('message', handler)
      })

      worker.postMessage({ type: 'init' })
      this.workers.push(worker)
      workerPromises.push(readyPromise)
    }

    await Promise.all(workerPromises)
    this.ready = true
  }

  /**
   * Run expectimax search for the current draft state.
   * Returns the final results (best depth completed).
   */
  async search(
    state: DraftState,
    data: DraftData,
    config?: Partial<ExpectimaxConfig>,
  ): Promise<ExpectimaxResult[]> {
    if (!this.ready) await this.init()

    const cfg = { ...DEFAULT_CONFIG, ...config }
    const searchState = createSearchState(state)
    const mySearchId = ++this.searchId

    // Prefilter candidates on main thread (fast, synchronous)
    const ban = isBanPhase(searchState)
    const ours = isOurTurn(searchState)
    const width = ban
      ? (ours ? cfg.ourBanWidth : cfg.oppBanWidth)
      : (ours ? cfg.ourPickWidth : cfg.oppPickWidth)

    const candidates = ban
      ? prefilterBans(searchState, data, width)
      : prefilterPicks(searchState, data, width)

    if (candidates.length === 0) return []

    // Split candidates across workers (round-robin)
    const workerCandidates: string[][] = Array.from({ length: this.workers.length }, () => [])
    for (let i = 0; i < candidates.length; i++) {
      workerCandidates[i % this.workers.length].push(candidates[i])
    }

    // Prepare serializable state (Sets don't survive structured clone)
    const serializableState = {
      ...searchState,
      taken: Array.from(searchState.taken),
    }

    // Collect results per depth
    const depthResults = new Map<number, ExpectimaxResult[]>()
    let finalResults: ExpectimaxResult[] = []

    return new Promise<ExpectimaxResult[]>((resolve) => {
      let workersCompleted = 0
      const activeWorkers = workerCandidates.filter(c => c.length > 0).length

      if (activeWorkers === 0) {
        resolve([])
        return
      }

      for (let i = 0; i < this.workers.length; i++) {
        if (workerCandidates[i].length === 0) continue

        const handler = (e: MessageEvent) => {
          // Ignore results from stale searches
          if (this.searchId !== mySearchId) {
            this.workers[i].removeEventListener('message', handler)
            return
          }

          if (e.data.type === 'depth-complete') {
            const { results, depth } = e.data
            const existing = depthResults.get(depth) || []
            depthResults.set(depth, [...existing, ...results])

            // Check if all workers have reported for this depth
            const allForDepth = depthResults.get(depth)!
            // Report progress: merge and sort all results for this depth
            const merged = [...allForDepth].sort((a: ExpectimaxResult, b: ExpectimaxResult) => b.score - a.score)
            this.onProgress?.(merged, depth)
          }

          if (e.data.type === 'result') {
            const { results } = e.data
            finalResults = [...finalResults, ...results]
            workersCompleted++

            if (workersCompleted >= activeWorkers) {
              this.workers[i].removeEventListener('message', handler)
              finalResults.sort((a, b) => b.score - a.score)
              resolve(finalResults)
            }
          }

          if (e.data.type === 'error') {
            console.error(`Expectimax worker error: ${e.data.message}`)
            workersCompleted++
            if (workersCompleted >= activeWorkers) {
              this.workers[i].removeEventListener('message', handler)
              resolve(finalResults.sort((a, b) => b.score - a.score))
            }
          }
        }

        this.workers[i].addEventListener('message', handler)
        this.workers[i].postMessage({
          type: 'search',
          rootState: serializableState,
          draftData: data,
          config: cfg,
          candidates: workerCandidates[i],
        })
      }
    })
  }

  /** Cancel ongoing search (workers will ignore stale results) */
  cancel(): void {
    this.searchId++
  }

  /** Terminate all workers */
  dispose(): void {
    for (const w of this.workers) w.terminate()
    this.workers = []
    this.ready = false
  }
}
