import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { noCharSeries } from '@readaloudkit/gop'
import { describe, expect, it } from 'vitest'
import type { GopWord, ProsodyFeatures, ScoringInput } from '@readaloudkit/types'
import { UTTERANCE_FEATURE_KEYS } from '@readaloudkit/features'
import {
  ATTENTION_THRESHOLD,
  CommunityScoringBackend,
  NoopScoringBackend,
  defaultScoringDir,
  loadScoringBackend,
  scoringReady,
  splitHiddenKeys,
} from './index.ts'

function gopWord(over: Partial<GopWord> = {}): GopWord {
  return {
    tok: 'word',
    t0: 0,
    t1: 0.3,
    f0: 0,
    f1: 15,
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

/**
 * A stand-in for the pooled acoustic layer.
 *
 * The values are arbitrary -- these cases check that the release loads, that
 * every word gets a band, and that the two readings of the threshold agree,
 * none of which depends on the audio. What it does have to get right is the
 * shape, since that is what the projection and the head widths are checked
 * against.
 */
function hidden(nWords: number, size = 768) {
  const vec = (seed: number) =>
    Float32Array.from({ length: size }, (_, i) => Math.sin(seed + i) * 0.1)
  return {
    words: Array.from({ length: nWords }, (_, i) => vec(i + 1)),
    utterance: vec(0),
    size,
    layer: null,
  }
}

function input(words: GopWord[]): ScoringInput {
  return {
    durationMs: 3000,
    words: [],
    gopWords: words,
    hypothesis: '',
    reference: '',
    prosody,
    hidden: hidden(words.length),
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

  it('refuses to score a release that needs a hidden layer without one', async () => {
    // Filling zeros would return numbers that look like scores. Whether this
    // case can arise at all depends on the shipped release, so it asserts the
    // matching behaviour either way rather than skipping.
    const backend = await CommunityScoringBackend.load()
    const withoutHidden = { ...input([gopWord()]), hidden: undefined }
    const manifest = JSON.parse(
      readFileSync(resolve(defaultScoringDir(), 'scoring.json'), 'utf8'),
    ) as { hiddenProjection?: unknown }
    if (manifest.hiddenProjection) {
      await expect(backend.score(withoutHidden)).rejects.toThrow(/hidden layer/)
    } else {
      await expect(backend.score(withoutHidden)).resolves.not.toBeNull()
    }
  })

  it('rejects a hidden layer of the wrong width', async () => {
    const backend = await CommunityScoringBackend.load()
    const wrong = { ...input([gopWord()]), hidden: hidden(1, 256) }
    const manifest = JSON.parse(
      readFileSync(resolve(defaultScoringDir(), 'scoring.json'), 'utf8'),
    ) as { hiddenProjection?: { size: number } }
    if (manifest.hiddenProjection) {
      await expect(backend.score(wrong)).rejects.toThrow(/256-dimensional/)
    }
  })
})

describe('attention threshold', () => {
  it('is the calibrated operating point, not a coin flip', () => {
    expect(ATTENTION_THRESHOLD).toBeGreaterThan(0.5)
    expect(ATTENTION_THRESHOLD).toBeLessThan(1)
  })
})

describe('hidden feature keys', () => {
  it('passes through a key list with no hidden features', () => {
    expect(splitHiddenKeys(['gopMean', 'dur'], 'x')).toEqual({
      plain: ['gopMean', 'dur'],
      hidden: 0,
    })
  })

  it('splits a contiguous hid tail off the end', () => {
    expect(splitHiddenKeys(['gopMean', 'hid0', 'hid1'], 'x')).toEqual({
      plain: ['gopMean'],
      hidden: 2,
    })
  })

  it('rejects hidden keys that are out of order', () => {
    // The order is the model input contract. Accepting hid1 before hid0 would
    // feed every component into the column of another one, which scores
    // nonsense rather than failing.
    expect(() => splitHiddenKeys(['gopMean', 'hid1', 'hid0'], 'word head')).toThrow(/contiguous/)
  })

  it('rejects a plain key hiding among the hidden tail', () => {
    expect(() => splitHiddenKeys(['hid0', 'dur', 'hid1'], 'word head')).toThrow(/contiguous/)
  })
})
