import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { noCharSeries } from '@readaloudkit/gop'
import { describe, expect, it } from 'vitest'
import type { GopWord, ProsodyFeatures, ScoringInput } from '@readaloudkit/types'
import { UTTERANCE_FEATURE_KEYS } from '@readaloudkit/features'
import {
  ATTENTION_THRESHOLD,
  CommunityScoringBackend,
  NoopScoringBackend,
  loadScoringBackend,
  scoringReady,
} from './index.ts'

function gopWord(over: Partial<GopWord> = {}): GopWord {
  return {
    tok: 'word',
    t0: 0,
    t1: 0.3,
    gopMean: -0.1,
    gopMin: -0.2,
    gopStd: 0.05,
    marginMean: 3,
    marginMin: 1,
    nFrames: 12,
    nChars: 4,
    dur: 0.3,
    charPerSec: 13.3,
    blankRatio: 0.2,
    logNFrames: Math.log(13),
    ...noCharSeries(),
    ...over,
  }
}

const prosody = Object.fromEntries(
  UTTERANCE_FEATURE_KEYS.map((k) => [k, 0]),
) as unknown as ProsodyFeatures

function input(words: GopWord[]): ScoringInput {
  return {
    durationMs: 3000,
    words: [],
    gopWords: words,
    hypothesis: '',
    reference: '',
    prosody,
    content: {
      score: 5,
      maxScore: 5,
      mode: 'calibrated',
      strict: 5,
      calibrated: 5,
      referenceWords: words.length,
      chargedErrors: 0,
    },
  }
}

describe('noop backend', () => {
  it('returns nothing so the rule-derived fields still stand alone', async () => {
    expect(await new NoopScoringBackend().score(input([gopWord()]))).toBeNull()
  })

  it('is what resolves when no heads are installed', async () => {
    const backend = await loadScoringBackend('/nonexistent/scoring')
    expect(backend.name).toBe('none')
  })
})

/**
 * The repository ships a release, so resolution must succeed here.
 *
 * These three cases are the only ones that load the weights that actually go
 * out, and skipping them is indistinguishable from passing them in a CI
 * summary. So the skip is allowed exactly once — for someone who deleted
 * `releases/` to use their own backend — and is a failure otherwise.
 */
const shipsARelease = existsSync(resolve(import.meta.dirname, '../../../releases/CURRENT'))
const installed = scoringReady()

describe('shipped release', () => {
  it('resolves to loadable heads whenever one is checked in', () => {
    if (!shipsARelease) return
    expect(installed).toBe(true)
  })
})

const withHeads = installed ? describe : describe.skip

withHeads('community backend', () => {
  it('bands every word and never contradicts its own flag', async () => {
    const backend = await CommunityScoringBackend.load()
    const words = [gopWord(), gopWord({ gopMean: -8, marginMean: -4 }), gopWord({ tok: 'x' })]
    const result = await backend.score(input(words))
    expect(result).not.toBeNull()
    expect(result!.backend).toBe('community')
    expect(result!.words).toHaveLength(words.length)
    for (const w of result!.words!) {
      // A word shown as anything but `good` must also be flagged, and vice
      // versa; both readings come off one threshold.
      expect(w.needsAttention).toBe(w.level !== 'good')
    }
  })

  it('reports utterance scores on the display range', async () => {
    const backend = await CommunityScoringBackend.load()
    const result = await backend.score(input([gopWord(), gopWord()]))
    for (const value of [result!.pronunciation, result!.fluency, result!.overall]) {
      expect(value).toBeGreaterThanOrEqual(10)
      expect(value).toBeLessThanOrEqual(90)
    }
  })

  it('handles an utterance with no aligned words', async () => {
    const backend = await CommunityScoringBackend.load()
    const result = await backend.score(input([]))
    expect(result).toEqual({ backend: 'community' })
  })
})

describe('attention threshold', () => {
  it('is the calibrated operating point, not a coin flip', () => {
    expect(ATTENTION_THRESHOLD).toBeGreaterThan(0.5)
    expect(ATTENTION_THRESHOLD).toBeLessThan(1)
  })
})
