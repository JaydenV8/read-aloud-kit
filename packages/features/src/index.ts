import type { GopWord, PauseWhere, ProsodyFeatures } from '@readaloudkit/types'

/**
 * Model input contract for utterance-level heads. A head is trained on this
 * exact sequence, followed by `UTTERANCE_GOP_KEYS` from `@readaloudkit/gop`.
 */
export const UTTERANCE_FEATURE_KEYS = [
  'duration',
  'nRef',
  'nAligned',
  'coverage',
  'spokenSec',
  'wpm',
  'nPause',
  'pauseTotal',
  'pauseMax',
  'pauseMean',
  'pauseRatio',
  'wordDurMean',
  'wordDurStd',
  'wordDurCv',
  'leadSil',
  'tailSil',
  'alignOk',
  'rmsMean',
  'rmsStd',
  'rmsPeak',
  'activeRatio',
] as const satisfies readonly (keyof ProsodyFeatures)[]

/** Shortest mid-utterance gap that counts as a pause for the feature vector. */
export const PAUSE_MIN = 0.15
/** A leading or trailing silence shorter than this is not worth reporting. */
export const EDGE_SIL_MIN = 0.05

const WORD_RE = /[A-Za-z']+/g

/** Pause spans in seconds, the unit the feature vector is defined in. */
export type TimedPause = { t0: number; t1: number; where: PauseWhere }

export type AcousticFeatures = Pick<
  ProsodyFeatures,
  'rmsMean' | 'rmsStd' | 'rmsPeak' | 'activeRatio'
>

export function countReferenceWords(text: string): number {
  return (text.match(WORD_RE) ?? []).length
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0
}

/** Sample standard deviation (n-1), matching the utterance feature definition. */
function sampleStd(xs: number[]): number {
  if (xs.length < 2) return 0
  const m = mean(xs)
  let acc = 0
  for (const x of xs) acc += (x - m) ** 2
  return Math.sqrt(acc / (xs.length - 1))
}

/**
 * Leading, mid and trailing silences around the aligned words. Mid gaps use a
 * fixed floor here; the adaptive, per-utterance threshold used for display
 * lives in `@readaloudkit/prosody`.
 */
export function pausesFromWords(words: GopWord[], duration: number): TimedPause[] {
  const timed = words.filter((w) => w.t0 != null && w.t1 != null)
  if (!timed.length) return []
  const out: TimedPause[] = []
  const first = timed[0]!
  if (first.t0 > EDGE_SIL_MIN) out.push({ t0: 0, t1: first.t0, where: 'lead' })
  for (let i = 0; i + 1 < timed.length; i++) {
    const a = timed[i]!
    const b = timed[i + 1]!
    if (b.t0 - a.t1 >= PAUSE_MIN) out.push({ t0: a.t1, t1: b.t0, where: 'mid' })
  }
  const last = timed[timed.length - 1]!
  const tail = duration - last.t1
  if (tail > EDGE_SIL_MIN)
    out.push({ t0: last.t1, t1: Math.round(duration * 1000) / 1000, where: 'tail' })
  return out
}

export function prosodyFromAlignment(input: {
  words: GopWord[]
  pauses: TimedPause[]
  referenceText: string
  duration: number
  alignOk?: boolean
  acoustic?: AcousticFeatures
  /**
   * Reference words the speaker actually said. Forced alignment lands every
   * reference word somewhere on the path, so the aligned count alone always
   * reports full coverage; pass the omission-aware count to get a real one.
   */
  spokenWords?: number
}): ProsodyFeatures {
  const words = input.words.filter((w) => w.t0 != null && w.t1 != null)
  const nRef = countReferenceWords(input.referenceText)
  const nAligned = words.length
  const durs = words.map((w) => Math.max(0, w.t1 - w.t0))
  const midPauses = input.pauses.filter((p) => p.where === 'mid' && p.t1 - p.t0 >= PAUSE_MIN)
  const pauseLens = midPauses.map((p) => p.t1 - p.t0)
  const spoken = durs.reduce((a, b) => a + b, 0)
  const durMean = mean(durs)
  const pauseTotal = pauseLens.reduce((a, b) => a + b, 0)
  const lead = input.pauses.find((p) => p.where === 'lead')
  const tail = input.pauses.find((p) => p.where === 'tail')
  const spokenWords = Math.max(0, Math.min(nRef, input.spokenWords ?? nAligned))

  return {
    duration: input.duration,
    nRef,
    nAligned,
    coverage: nRef ? spokenWords / nRef : 0,
    spokenSec: spoken,
    wpm: spoken > 0.2 ? nAligned / (spoken / 60) : 0,
    nPause: midPauses.length,
    pauseTotal,
    pauseMax: pauseLens.length ? Math.max(...pauseLens) : 0,
    pauseMean: mean(pauseLens),
    pauseRatio: input.duration ? pauseTotal / input.duration : 0,
    wordDurMean: durMean,
    wordDurStd: sampleStd(durs),
    wordDurCv: durMean > 1e-6 ? sampleStd(durs) / durMean : 0,
    leadSil: lead ? lead.t1 - lead.t0 : 0,
    tailSil: tail ? tail.t1 - tail.t0 : 0,
    alignOk: input.alignOk === false ? 0 : 1,
    rmsMean: input.acoustic?.rmsMean ?? 0,
    rmsStd: input.acoustic?.rmsStd ?? 0,
    rmsPeak: input.acoustic?.rmsPeak ?? 0,
    activeRatio: input.acoustic?.activeRatio ?? 0,
  }
}

/** Frame energy statistics over the raw 16 kHz signal. */
export function acousticFeatures(samples: Float32Array, sampleRate = 16000): AcousticFeatures {
  const empty: AcousticFeatures = { rmsMean: 0, rmsStd: 0, rmsPeak: 0, activeRatio: 0 }
  if (!samples.length) return empty
  const hop = Math.max(1, Math.trunc(sampleRate * 0.02))
  const nFrames = Math.trunc(samples.length / hop)
  if (nFrames === 0) return empty

  let sumSq = 0
  let sum = 0
  let peak = 0
  for (const x of samples) {
    sumSq += x * x
    sum += x
    const a = Math.abs(x)
    if (a > peak) peak = a
  }
  const avg = sum / samples.length
  let varSum = 0
  for (const x of samples) varSum += (x - avg) ** 2

  const energies = new Float64Array(nFrames)
  for (let f = 0; f < nFrames; f++) {
    let acc = 0
    const off = f * hop
    for (let i = 0; i < hop; i++) {
      const x = samples[off + i]!
      acc += x * x
    }
    energies[f] = Math.sqrt(acc / hop + 1e-12)
  }
  const sorted = Array.from(energies).sort((a, b) => a - b)
  const mid = sorted.length >> 1
  const medianEnergy = sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2
  const thr = medianEnergy * 0.3
  let active = 0
  for (const e of energies) if (e > thr) active += 1

  return {
    rmsMean: Math.sqrt(sumSq / samples.length),
    rmsStd: Math.sqrt(varSum / samples.length),
    rmsPeak: peak,
    activeRatio: active / nFrames,
  }
}

/** Feature vector for an utterance-level head, in `UTTERANCE_FEATURE_KEYS` order. */
export function utteranceVector(f: ProsodyFeatures): number[] {
  return UTTERANCE_FEATURE_KEYS.map((k) => {
    const v = f[k]
    return typeof v === 'number' && Number.isFinite(v) ? v : 0
  })
}
