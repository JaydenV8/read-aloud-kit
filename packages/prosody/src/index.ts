import type { GopWord } from '@readaloudkit/types'

/** Absolute floor for the adaptive mid-gap threshold, in seconds. */
export const PAUSE_FLOOR = 0.15
/** Multiplier on the within-utterance gap spread. */
export const PAUSE_K = 1
/** Adaptive gaps shorter than this are real but not worth showing a reader. */
export const PAUSE_DISPLAY_MIN = 0.38
/** Gaps below this are inter-word coarticulation, not candidate pauses. */
export const GAP_MIN = 0.05

export type Gap = { t0: number; t1: number }

function populationStd(xs: number[]): number {
  if (!xs.length) return 0
  const m = xs.reduce((a, b) => a + b, 0) / xs.length
  let acc = 0
  for (const x of xs) acc += (x - m) ** 2
  return Math.sqrt(acc / xs.length)
}

function median(xs: number[]): number {
  if (!xs.length) return 0
  const sorted = [...xs].sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2
}

/** Every gap between consecutive aligned words, longest-first ordering not applied. */
export function midGaps(words: GopWord[], minDur = 0): Gap[] {
  const timed = words.filter((w) => w.t0 != null && w.t1 != null).sort((a, b) => a.t0 - b.t0)
  const out: Gap[] = []
  for (let i = 0; i + 1 < timed.length; i++) {
    const a = timed[i]!
    const b = timed[i + 1]!
    if (b.t0 - a.t1 >= minDur) out.push({ t0: a.t1, t1: b.t0 })
  }
  return out
}

/**
 * Keep the gaps that are long *relative to this utterance*. A slow speaker
 * pauses everywhere and a fast one barely at all, so a fixed threshold either
 * floods or misses; the median plus one spread adapts to the speaker while the
 * floor stops a metronomic reader from having every gap flagged.
 */
export function adaptiveGaps(words: GopWord[], floor = PAUSE_FLOOR, k = PAUSE_K): Gap[] {
  const raw = midGaps(words, GAP_MIN)
  if (!raw.length) return []
  const lens = raw.map((g) => g.t1 - g.t0)
  const thr = Math.max(floor, median(lens) + k * populationStd(lens))
  return raw.filter((g) => g.t1 - g.t0 >= thr)
}

/**
 * Drop gaps that are not actually quiet. The aligner can leave a hole where the
 * speaker was still making noise (breath, filler, a mistracked word).
 */
export function energyFilter(
  gaps: Gap[],
  samples: Float32Array,
  sampleRate: number,
  maxRatio: number,
): Gap[] {
  if (!samples.length || maxRatio <= 0) return gaps
  const abs = Array.from(samples, Math.abs).sort((a, b) => a - b)
  const mid = abs.length >> 1
  const med = (abs.length % 2 ? abs[mid]! : (abs[mid - 1]! + abs[mid]!) / 2) + 1e-8
  return gaps.filter((g) => {
    let i0 = Math.trunc(g.t0 * sampleRate)
    let i1 = Math.trunc(g.t1 * sampleRate)
    i0 = Math.max(0, Math.min(i0, samples.length))
    i1 = Math.max(i0 + 1, Math.min(i1, samples.length))
    let acc = 0
    for (let i = i0; i < i1; i++) acc += samples[i]! * samples[i]!
    return Math.sqrt(acc / (i1 - i0)) <= maxRatio * med
  })
}

/** Adaptive gaps that are also long enough to be worth surfacing to a reader. */
export function displayPauses(
  words: GopWord[],
  opts: { floor?: number; k?: number; minDisplay?: number } = {},
): Gap[] {
  const minDisplay = opts.minDisplay ?? PAUSE_DISPLAY_MIN
  return adaptiveGaps(words, opts.floor ?? PAUSE_FLOOR, opts.k ?? PAUSE_K).filter(
    (g) => g.t1 - g.t0 >= minDisplay,
  )
}
