import { noCharSeries } from '@readaloudkit/gop'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { GopWord } from '@readaloudkit/types'
import { GAP_MIN, adaptiveGaps, displayPauses, energyFilter, midGaps } from './index.ts'

type Fixture = {
  duration: number
  words: { tok: string; t0: number; t1: number }[]
  mid_gaps: [number, number][]
  adaptive_gaps: [number, number][]
}

const fixture = JSON.parse(
  readFileSync(resolve(import.meta.dirname, '../../../tests/fixtures/features.json'), 'utf8'),
) as Fixture

function asGopWord(w: { tok: string; t0: number; t1: number }): GopWord {
  return {
    tok: w.tok,
    t0: w.t0,
    t1: w.t1,
    f0: Math.round(w.t0 * 50),
    f1: Math.round(w.t1 * 50),
    gopMean: 0,
    gopMin: 0,
    gopStd: 0,
    marginMean: 0,
    marginMin: 0,
    nFrames: 0,
    nChars: w.tok.length,
    dur: w.t1 - w.t0,
    charPerSec: 0,
    blankRatio: 0,
    logNFrames: 0,
    ...noCharSeries(),
  }
}

const words = fixture.words.map(asGopWord)

describe('pause detection vs python reference', () => {
  it('collects the same candidate gaps', () => {
    const got = midGaps(words, GAP_MIN).map((g) => [g.t0, g.t1])
    expect(got).toEqual(fixture.mid_gaps)
  })

  it('keeps the same gaps after the adaptive threshold', () => {
    const got = adaptiveGaps(words, 0.15, 1).map((g) => [g.t0, g.t1])
    expect(got).toEqual(fixture.adaptive_gaps)
  })

  it('keeps the gap that stands out and drops the ordinary ones', () => {
    // gaps 0.1 / 0.1 / 0.8: only the last clears median + one spread
    const uneven = [
      asGopWord({ tok: 'a', t0: 0.0, t1: 0.4 }),
      asGopWord({ tok: 'b', t0: 0.5, t1: 0.9 }),
      asGopWord({ tok: 'c', t0: 1.0, t1: 1.4 }),
      asGopWord({ tok: 'd', t0: 2.2, t1: 2.6 }),
    ]
    expect(adaptiveGaps(uneven, 0.15, 1).map((g) => [g.t0, g.t1])).toEqual([[1.4, 2.2]])
  })

  it('flags every gap when the reader breaks after each word', () => {
    // A uniformly chopped delivery has zero spread, so the threshold collapses
    // to the median and each gap clears it. That is the intended reading: the
    // problem is the chopping, not one outlier.
    const chopped = [0, 1, 2, 3].map((i) => asGopWord({ tok: 'w', t0: i, t1: i + 0.5 }))
    expect(adaptiveGaps(chopped, 0.15, 1)).toHaveLength(3)
  })

  it('drops display pauses that are real but too short to show', () => {
    expect(displayPauses(words)).toEqual([{ t0: 0.9, t1: 1.6 }])
    expect(displayPauses(words, { minDisplay: 5 })).toEqual([])
  })

  it('discards gaps the speaker was still making noise through', () => {
    const sr = 16000
    const loud = new Float32Array(sr * 2)
    loud.fill(0.5)
    const gaps = [{ t0: 0.9, t1: 1.6 }]
    expect(energyFilter(gaps, loud, sr, 0.5)).toEqual([])
    expect(energyFilter(gaps, loud, sr, 2)).toEqual(gaps)
    expect(energyFilter(gaps, new Float32Array(0), sr, 0.5)).toEqual(gaps)
  })
})
