import { describe, expect, it } from 'vitest'
import type { GopChar, GopWord } from '@readaloudkit/types'
import {
  CHAR_FEATURE_KEYS,
  UTTERANCE_GOP_KEYS,
  UTTERANCE_GOP_KEYS_V2,
  WEAK_CHAR_GOP,
  WORD_FEATURE_KEYS,
  WORD_FEATURE_KEYS_V2,
  addContext,
  charFeatures,
  noCharSeries,
  utteranceGopFeatures,
  utteranceGopFeaturesV2,
  wordVectorV2,
} from './index.ts'

const ch = (gop: number, dur: number, margin = 5): GopChar => ({ lab: 'x', gop, margin, dur })

function word(over: Partial<GopWord> = {}): GopWord {
  return {
    tok: 'word',
    t0: 0,
    t1: 0.3,
    gopMean: -1,
    gopMin: -2,
    gopStd: 0.5,
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

describe('feature contracts', () => {
  it('keeps v1 as an exact prefix of v2', () => {
    expect(WORD_FEATURE_KEYS_V2.slice(0, WORD_FEATURE_KEYS.length)).toEqual([...WORD_FEATURE_KEYS])
    expect(UTTERANCE_GOP_KEYS_V2.slice(0, UTTERANCE_GOP_KEYS.length)).toEqual([
      ...UTTERANCE_GOP_KEYS,
    ])
  })

  it('has no duplicate keys', () => {
    expect(new Set(WORD_FEATURE_KEYS_V2).size).toBe(WORD_FEATURE_KEYS_V2.length)
    expect(new Set(UTTERANCE_GOP_KEYS_V2).size).toBe(UTTERANCE_GOP_KEYS_V2.length)
  })

  it('vectorises to the declared width', () => {
    expect(wordVectorV2(word())).toHaveLength(WORD_FEATURE_KEYS_V2.length)
    expect(WORD_FEATURE_KEYS_V2).toHaveLength(WORD_FEATURE_KEYS.length + CHAR_FEATURE_KEYS.length)
  })

  it('substitutes a finite value for a non-finite v2 feature', () => {
    expect(wordVectorV2(word({ durFrontBackRatio: Infinity }))).not.toContain(Infinity)
    expect(wordVectorV2(word({ charDurCv: NaN })).some(Number.isNaN)).toBe(false)
  })
})

describe('charFeatures', () => {
  it('returns zeros for an empty series', () => {
    expect(Object.values(charFeatures([])).every((v) => v === 0)).toBe(true)
  })

  it('normalises positions so word length does not change the description', () => {
    // Worst character last in both, so both should report position 1.
    const short = charFeatures([ch(-1, 0.1), ch(-9, 0.1)])
    const long = charFeatures([ch(-1, 0.1), ch(-1, 0.1), ch(-1, 0.1), ch(-9, 0.1)])
    expect(short.worstCharPos).toBe(1)
    expect(long.worstCharPos).toBe(1)
  })

  it('separates two words a word-level average cannot tell apart', () => {
    // Same mean posterior, -2. One read evenly, one with a mangled character.
    const even = charFeatures([ch(-2, 0.1), ch(-2, 0.1), ch(-2, 0.1), ch(-2, 0.1)])
    const spiky = charFeatures([ch(-0.2, 0.1), ch(-0.2, 0.1), ch(-0.2, 0.1), ch(-7.4, 0.1)])
    expect(even.charGopRange).toBeCloseTo(0, 10)
    expect(spiky.charGopRange).toBeCloseTo(7.2, 10)
    expect(spiky.charGopMin).toBeLessThan(even.charGopMin)
  })

  it('reads stress off the duration series', () => {
    // A long first character against a long last character: same mean duration,
    // opposite stress. Only the shape features can see it.
    const front = charFeatures([ch(-1, 0.3), ch(-1, 0.1), ch(-1, 0.1), ch(-1, 0.1)])
    const back = charFeatures([ch(-1, 0.1), ch(-1, 0.1), ch(-1, 0.1), ch(-1, 0.3)])
    expect(front.longestCharPos).toBe(0)
    expect(back.longestCharPos).toBe(1)
    expect(front.durFrontBackRatio).toBeGreaterThan(1)
    expect(back.durFrontBackRatio).toBeLessThan(1)
    expect(front.charDurCv).toBeCloseTo(back.charDurCv, 10)
  })

  it('compares the front of a word against its back', () => {
    const decaying = charFeatures([ch(-0.5, 0.1), ch(-0.5, 0.1), ch(-4, 0.1), ch(-4, 0.1)])
    expect(decaying.headGop).toBeGreaterThan(decaying.tailGop)
    expect(decaying.headTailDelta).toBeGreaterThan(0)
  })

  it('counts weak characters against the corpus-derived threshold', () => {
    const f = charFeatures([ch(WEAK_CHAR_GOP - 0.1, 0.1), ch(WEAK_CHAR_GOP + 0.1, 0.1)])
    expect(f.nWeakChars).toBe(1)
    expect(f.fracWeakChars).toBe(0.5)
  })
})

describe('addContext', () => {
  const build = () => [
    word({ tok: 'a', t0: 0, t1: 0.2, dur: 0.2, gopMean: -1, charPerSec: 10 }),
    word({ tok: 'b', t0: 0.9, t1: 1.1, dur: 0.2, gopMean: -3, charPerSec: 10 }),
    word({ tok: 'c', t0: 1.2, t1: 1.8, dur: 0.6, gopMean: -5, charPerSec: 20 }),
  ]

  it('measures the silence either side of a word', () => {
    const ws = build()
    addContext(ws)
    expect(ws[0]!.gapBefore).toBe(0)
    expect(ws[1]!.gapBefore).toBeCloseTo(0.7, 10)
    expect(ws[0]!.gapAfter).toBeCloseTo(0.7, 10)
    expect(ws[2]!.gapAfter).toBe(0)
  })

  it('carries the neighbours posteriors', () => {
    const ws = build()
    addContext(ws)
    expect(ws[1]!.prevGopMean).toBe(-1)
    expect(ws[1]!.nextGopMean).toBe(-5)
    expect(ws[0]!.prevGopMean).toBe(0)
    expect(ws[2]!.nextGopMean).toBe(0)
  })

  it('expresses duration relative to the utterance, not in seconds', () => {
    const ws = build()
    addContext(ws)
    // mean duration is (0.2 + 0.2 + 0.6) / 3 = 1/3
    expect(ws[0]!.relDur).toBeCloseTo(0.6, 10)
    expect(ws[2]!.relDur).toBeCloseTo(1.8, 10)
  })

  it('leaves an empty utterance alone', () => {
    expect(() => addContext([])).not.toThrow()
  })
})

describe('utteranceGopFeaturesV2', () => {
  it('returns zeros for an empty utterance', () => {
    const f = utteranceGopFeaturesV2([])
    expect(UTTERANCE_GOP_KEYS_V2.every((k) => f[k] === 0)).toBe(true)
  })

  it('reproduces the v1 summary exactly on the keys they share', () => {
    // This is what lets a v2 utterance feed a head trained on the v1 contract.
    const ws = [-4, -3, -2, -1].map((g) => word({ gopMean: g, marginMean: g }))
    const v1 = utteranceGopFeatures(ws)
    const v2 = utteranceGopFeaturesV2(ws)
    for (const k of UTTERANCE_GOP_KEYS) expect(v2[k]).toBe(v1[k])
  })

  it('tells a uniformly weak reader from one who fumbled two words', () => {
    const decayEverywhere = [0, 1, 2, 3].map(() =>
      word({ ...charFeatures([ch(-0.5, 0.1), ch(-4, 0.1)]) }),
    )
    const mostlyClean = [
      word({ ...charFeatures([ch(-0.5, 0.1), ch(-0.5, 0.1)]) }),
      word({ ...charFeatures([ch(-0.5, 0.1), ch(-0.5, 0.1)]) }),
      word({ ...charFeatures([ch(-9, 0.1), ch(-9, 0.1)]) }),
      word({ ...charFeatures([ch(-0.5, 0.1), ch(-0.5, 0.1)]) }),
    ]
    const a = utteranceGopFeaturesV2(decayEverywhere)
    const b = utteranceGopFeaturesV2(mostlyClean)
    // Every word decays toward its end in the first; none does in the second.
    expect(a.headTailDeltaMean).toBeGreaterThan(0.5)
    expect(b.headTailDeltaMean).toBeCloseTo(0, 10)
    // The second has a worse single word despite a cleaner average.
    expect(b.charGopMinP10).toBeLessThan(a.charGopMinP10)
  })
})
