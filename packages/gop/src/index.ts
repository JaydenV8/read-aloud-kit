import type { GopWord } from '@readaloudkit/types'

/**
 * Model input contract for word-level heads. Order matters: a scoring head is
 * trained on this exact sequence.
 */
export const WORD_FEATURE_KEYS = [
  'gopMean',
  'gopMin',
  'gopStd',
  'marginMean',
  'marginMin',
  'nFrames',
  'nChars',
  'dur',
  'charPerSec',
  'blankRatio',
  'logNFrames',
] as const satisfies readonly (keyof GopWord)[]

/** Utterance-level GOP aggregates, appended after the prosody features. */
export const UTTERANCE_GOP_KEYS = [
  'gopMean',
  'gopStd',
  'gopMin',
  'gopP10',
  'gopP25',
  'marginMean',
  'marginP10',
  'nGopWords',
  'fracLowGop',
] as const

export type UtteranceGopFeatures = Record<(typeof UTTERANCE_GOP_KEYS)[number], number>

function stats(xs: number[]): [number, number, number] {
  if (!xs.length) return [0, 0, 0]
  let sum = 0
  let min = Infinity
  for (const x of xs) {
    sum += x
    if (x < min) min = x
  }
  const mean = sum / xs.length
  let varSum = 0
  for (const x of xs) varSum += (x - mean) ** 2
  return [mean, min, Math.sqrt(varSum / xs.length)]
}

/** numpy.median: average of the two central values on an even-length input. */
function median(sorted: number[]): number {
  const n = sorted.length
  if (!n) return 0
  const mid = n >> 1
  return n % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2
}

/** numpy.quantile with the default `linear` interpolation. */
function quantile(sorted: number[], q: number): number {
  const n = sorted.length
  if (!n) return 0
  if (n === 1) return sorted[0]!
  const pos = q * (n - 1)
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  if (lo === hi) return sorted[lo]!
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo)
}

export function wordsFromPath(
  path: number[],
  logProbs: Float32Array,
  frames: number,
  vocabSize: number,
  labels: string[],
  blank: number,
  duration: number,
): GopWord[] {
  const nFrames = path.length
  if (!nFrames) return []
  const secPer = duration / nFrames
  const tokens: { id: number; lab: string; frames: number[] }[] = []
  let prev: number | null = null
  for (let t = 0; t < path.length; t++) {
    const tid = path[t]!
    if (tid === blank) {
      prev = blank
      continue
    }
    if (prev === tid && tokens.length) {
      tokens[tokens.length - 1]!.frames.push(t)
    } else {
      tokens.push({ id: tid, lab: labels[tid] ?? '', frames: [t] })
    }
    prev = tid
  }

  const words: GopWord[] = []
  let cur: typeof tokens = []

  const flush = () => {
    if (!cur.length) return
    const frameList = cur.flatMap((tok) => tok.frames)
    if (!frameList.length) {
      cur = []
      return
    }
    const gops: number[] = []
    const margins: number[] = []
    for (const tok of cur) {
      for (const t of tok.frames) {
        const off = t * vocabSize
        const g = logProbs[off + tok.id]!
        gops.push(g)
        let other = -1e9
        for (let c = 0; c < vocabSize; c++) {
          if (c === tok.id) continue
          const v = logProbs[off + c]!
          if (v > other) other = v
        }
        margins.push(g - other)
      }
    }
    const first = frameList[0]!
    const last = frameList[frameList.length - 1]!
    const t0 = first * secPer
    const t1 = (last + 1) * secPer
    const span = Math.max(1, last - first + 1)
    let blankN = 0
    for (let t = first; t <= last; t++) {
      if (path[t] === blank) blankN += 1
    }
    const [gMean, gMin, gStd] = stats(gops)
    const [mMean, mMin] = stats(margins)
    const tok = cur.map((t) => t.lab).join('')
    const nChars = cur.length
    const dur = Math.max(1e-3, t1 - t0)
    words.push({
      tok,
      t0: Math.round(t0 * 1000) / 1000,
      t1: Math.round(t1 * 1000) / 1000,
      gopMean: gMean,
      gopMin: gMin,
      gopStd: gStd,
      marginMean: mMean,
      marginMin: mMin,
      nFrames: frameList.length,
      nChars,
      dur,
      charPerSec: nChars / dur,
      blankRatio: blankN / span,
      logNFrames: Math.log(frameList.length + 1),
    })
    cur = []
  }

  for (const tok of tokens) {
    if (tok.lab === '|') {
      flush()
      continue
    }
    cur.push(tok)
  }
  flush()
  return words
}

/** Feature vector for a word-level head, in `WORD_FEATURE_KEYS` order. */
export function wordVector(w: GopWord): number[] {
  return WORD_FEATURE_KEYS.map((k) => {
    const v = w[k]
    return typeof v === 'number' && Number.isFinite(v) ? v : 0
  })
}

export function utteranceGopFeatures(words: GopWord[]): UtteranceGopFeatures {
  if (!words.length) {
    return {
      gopMean: 0,
      gopStd: 0,
      gopMin: 0,
      gopP10: 0,
      gopP25: 0,
      marginMean: 0,
      marginP10: 0,
      nGopWords: 0,
      fracLowGop: 0,
    }
  }
  const g = words.map((w) => w.gopMean)
  const m = words.map((w) => w.marginMean)
  const gSorted = [...g].sort((a, b) => a - b)
  const mSorted = [...m].sort((a, b) => a - b)
  const [gMean, gMin, gStd] = stats(g)
  const [mMean] = stats(m)
  const cutoff = median(gSorted) - gStd
  let low = 0
  for (const x of g) if (x < cutoff) low += 1
  return {
    gopMean: gMean,
    gopStd: gStd,
    gopMin: gMin,
    gopP10: quantile(gSorted, 0.1),
    gopP25: quantile(gSorted, 0.25),
    marginMean: mMean,
    marginP10: quantile(mSorted, 0.1),
    nGopWords: words.length,
    fracLowGop: low / g.length,
  }
}

/**
 * Squashed posterior margin in (0, 1). 0.5 means the aligned character only
 * ties with its strongest competitor; near 0 means the frames do not support
 * the reference character at all.
 */
export function wordConfidence(w: GopWord): number {
  const x = 1 / (1 + Math.exp(-w.marginMean))
  return Math.max(0, Math.min(1, x))
}

/**
 * Mean per-frame log-posterior below which the aligner is treated as having
 * parked the reference word on silence rather than on speech.
 *
 * On the regression utterance, spoken words land between -0.001 and -3.47 while
 * the skipped word lands at -16.31, so the boundary is wide. It sits at -8
 * rather than mid-gap because the two errors are not symmetric: calling a
 * badly-pronounced word an omission moves a pronunciation problem onto the
 * content score, which is the one thing content must not measure. Missing a
 * skipped word only makes content slightly generous.
 *
 * This is calibrated on a handful of utterances. The principled fit is against
 * a corpus that scores words for pronunciation independently, which is what the
 * speechocean762 word-level labels provide.
 */
export const GOP_SILENT = -8

export type OmissionOptions = {
  /** Lower this to charge more omissions, raise it to charge fewer. */
  gopFloor?: number
}

/**
 * A reference word the alignment placed on almost no speech.
 *
 * The length-based clauses are the duration heuristics; they only fire for
 * words of five characters or more, which is why the `gopFloor` clause exists:
 * short words such as "many" are skipped just as often and carry the same
 * acoustic signature, but never trip a duration threshold.
 */
export function wordIsOmission(w: GopWord, opts: OmissionOptions = {}): boolean {
  const gopFloor = opts.gopFloor ?? GOP_SILENT
  const nChars = w.nChars || w.tok.trim().length || 0
  const { nFrames, dur, blankRatio } = w
  if (nChars <= 0) return true
  if (nFrames <= 1 && dur < 0.03) return true
  if (nChars >= 5 && (dur < 0.07 || nFrames <= 2)) return true
  if (nChars >= 8 && dur < 0.12) return true
  if (blankRatio >= 0.92 && nChars >= 5 && dur < 0.1) return true
  if (w.gopMean <= gopFloor) return true
  return false
}
