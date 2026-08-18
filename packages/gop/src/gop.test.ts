import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { GopWord } from '@readaloudkit/types'
import {
  GOP_SILENT,
  WORD_FEATURE_KEYS,
  noCharSeries,
  utteranceGopFeatures,
  wordConfidence,
  wordIsOmission,
  wordVector,
  wordsFromPath,
} from './index.ts'

type CatFixture = {
  path: number[]
  log_probs: number[][]
  gop_words: Array<Record<string, number | string>>
}

const gold = JSON.parse(
  readFileSync(resolve(import.meta.dirname, '../../../tests/fixtures/align_cat.json'), 'utf8'),
) as CatFixture

const labels = JSON.parse(
  readFileSync(resolve(import.meta.dirname, '../../../models/labels.json'), 'utf8'),
) as { labels: string[] }

/** python snake_case -> the field it pins on GopWord */
const FIELD_MAP: Record<string, keyof GopWord> = {
  t0: 't0',
  t1: 't1',
  gop_mean: 'gopMean',
  gop_min: 'gopMin',
  gop_std: 'gopStd',
  margin_mean: 'marginMean',
  margin_min: 'marginMin',
  n_frames: 'nFrames',
  n_chars: 'nChars',
  dur: 'dur',
  char_per_sec: 'charPerSec',
  blank_ratio: 'blankRatio',
  log_n_frames: 'logNFrames',
}

function word(over: Partial<GopWord> = {}): GopWord {
  return {
    tok: 'many',
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

describe('gop wordsFromPath', () => {
  const frames = gold.log_probs.length
  const vocab = gold.log_probs[0]!.length
  const logProbs = new Float32Array(frames * vocab)
  for (let t = 0; t < frames; t++) logProbs.set(gold.log_probs[t]!, t * vocab)
  const words = wordsFromPath(gold.path, logProbs, frames, vocab, labels.labels, 0, 1)

  it('matches python planted CAT words', () => {
    expect(words.map((w) => w.tok)).toEqual(gold.gop_words.map((w) => w.tok))
  })

  it('matches python on every model input field', () => {
    for (let i = 0; i < words.length; i++) {
      for (const [pyKey, tsKey] of Object.entries(FIELD_MAP)) {
        expect(words[i]![tsKey], `${gold.gop_words[i]!.tok}.${pyKey}`).toBeCloseTo(
          gold.gop_words[i]![pyKey] as number,
          5,
        )
      }
    }
  })

  it('vectorises in the trained feature order', () => {
    const v = wordVector(words[0]!)
    expect(v).toHaveLength(WORD_FEATURE_KEYS.length)
    expect(v[0]).toBeCloseTo(words[0]!.gopMean, 10)
    expect(v[WORD_FEATURE_KEYS.indexOf('blankRatio')]).toBeCloseTo(words[0]!.blankRatio, 10)
  })

  it('substitutes a finite value for a non-finite feature', () => {
    expect(wordVector(word({ charPerSec: Infinity }))).not.toContain(Infinity)
    expect(wordVector(word({ gopMean: NaN }))[0]).toBe(0)
  })
})

describe('wordConfidence', () => {
  it('spans the full range instead of saturating below a half', () => {
    expect(wordConfidence(word({ marginMean: 9 }))).toBeGreaterThan(0.99)
    expect(wordConfidence(word({ marginMean: 0 }))).toBeCloseTo(0.5, 10)
    expect(wordConfidence(word({ marginMean: -16 }))).toBeLessThan(0.001)
  })
})

describe('wordIsOmission', () => {
  it('accepts a normally spoken word', () => {
    expect(wordIsOmission(word())).toBe(false)
  })

  it('flags a long word squeezed into no time', () => {
    expect(wordIsOmission(word({ tok: 'university', nChars: 10, dur: 0.05 }))).toBe(true)
  })

  it('flags a short word the aligner parked on silence', () => {
    // The duration clauses only fire from five characters up, so "many" is only
    // caught by its posterior. This is the case the edit softening misses.
    const many = word({ tok: 'many', nChars: 4, dur: 0.08, gopMean: -16.3 })
    expect(wordIsOmission(many)).toBe(true)
    expect(wordIsOmission({ ...many, gopMean: -3.5 })).toBe(false)
  })

  it('leaves a badly pronounced word alone', () => {
    expect(wordIsOmission(word({ gopMean: GOP_SILENT + 1 }))).toBe(false)
  })

  it('honours a caller-supplied floor', () => {
    const w = word({ gopMean: -6 })
    expect(wordIsOmission(w)).toBe(false)
    expect(wordIsOmission(w, { gopFloor: -5 })).toBe(true)
  })
})

describe('utteranceGopFeatures', () => {
  it('returns zeros for an empty utterance', () => {
    const f = utteranceGopFeatures([])
    expect(Object.values(f).every((v) => v === 0)).toBe(true)
  })

  it('interpolates quantiles the way numpy does', () => {
    const ws = [-4, -3, -2, -1].map((g) => word({ gopMean: g, marginMean: g }))
    const f = utteranceGopFeatures(ws)
    expect(f.nGopWords).toBe(4)
    expect(f.gopMin).toBe(-4)
    expect(f.gopMean).toBeCloseTo(-2.5, 10)
    // linear interpolation at position 0.1*(4-1) = 0.3 between -4 and -3
    expect(f.gopP10).toBeCloseTo(-3.7, 10)
    expect(f.gopP25).toBeCloseTo(-3.25, 10)
  })
})
