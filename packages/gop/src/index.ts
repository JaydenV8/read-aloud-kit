import type { GopChar, GopWord } from '@readaloudkit/types'

/**
 * Model input contract for word-level heads. Order matters: a scoring head is
 * trained on this exact sequence.
 *
 * This is the v1 contract, eleven numbers that are all averages or extremes over
 * a word's frames. `WORD_FEATURE_KEYS_V2` keeps the shape of the per-character
 * series instead; see the model card for which contract a given head uses.
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

/**
 * What the per-character series adds on top of `WORD_FEATURE_KEYS`.
 *
 * The v1 eleven cannot tell a clean reading from one mangled syllable, and they
 * cannot represent stress at all, because stress is carried by how duration is
 * distributed *across* a word and a word-level average erases that by
 * construction. These describe where the worst character sits, how uneven the
 * durations are, and how the front of the word compares to the back — plus one
 * word of context either side.
 */
export const CHAR_FEATURE_KEYS = [
  'charGopMin',
  'charGopMax',
  'charGopStd',
  'charGopRange',
  'charGopP25',
  'worstCharPos',
  'nWeakChars',
  'fracWeakChars',
  'headGop',
  'tailGop',
  'headTailDelta',
  'charDurMax',
  'charDurMin',
  'charDurStd',
  'charDurCv',
  'longestCharPos',
  'durFrontBackRatio',
  'charMarginMin',
  'charMarginP25',
  'gapBefore',
  'gapAfter',
  'prevGopMean',
  'nextGopMean',
  'relDur',
  'relCharPerSec',
] as const satisfies readonly (keyof GopWord)[]

/** Model input contract for v2 word-level heads. */
export const WORD_FEATURE_KEYS_V2 = [...WORD_FEATURE_KEYS, ...CHAR_FEATURE_KEYS] as const

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

/**
 * Utterance summary that carries within-word shape upward.
 *
 * A speaker whose every word decays toward its end is not the same as one who
 * fumbles two words and reads the rest cleanly, even when the two utterances
 * average out identically. The first nine are `UTTERANCE_GOP_KEYS` unchanged, so
 * a head trained on this list can be compared against one trained on the short
 * list with no other difference.
 */
export const UTTERANCE_GOP_KEYS_V2 = [
  ...UTTERANCE_GOP_KEYS,
  'charGopMinMean',
  'charGopMinP10',
  'charGopStdMean',
  'fracWeakCharsMean',
  'fracWordsWithWeakChar',
  'headTailDeltaMean',
  'headTailDeltaStd',
  'charDurCvMean',
  'charDurCvStd',
  'worstCharPosMean',
  'longestCharPosMean',
  'relDurStd',
  'gapBeforeMax',
  'gapBeforeMean',
  'wordGopRange',
] as const

export type UtteranceGopFeaturesV2 = Record<(typeof UTTERANCE_GOP_KEYS_V2)[number], number>

/**
 * Mean log-posterior below which a character counts as weak.
 *
 * Swept by `training/sweep_weak_char.py` on the speechocean762 training split,
 * minus the speakers held out for early stopping, against whether the expert
 * marked the word down. The test split is never read by the sweep: a threshold
 * selected against it would leak it into every head downstream.
 *
 * The objective is flat between about -5 and -2.5 (F1 0.462 to 0.473), so the
 * exact value matters less than deriving it here. A different corpus should run
 * the sweep again rather than inherit this number.
 */
export const WEAK_CHAR_GOP = -4.3

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
    const chars: GopChar[] = []
    for (const tok of cur) {
      let cGopSum = 0
      let cMarginSum = 0
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
        cGopSum += g
        cMarginSum += g - other
      }
      const n = tok.frames.length
      chars.push({
        lab: tok.lab,
        gop: n ? cGopSum / n : 0,
        margin: n ? cMarginSum / n : 0,
        dur: n * secPer,
      })
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
      chars,
      ...charFeatures(chars),
      // Filled by addContext once every word exists.
      gapBefore: 0,
      gapAfter: 0,
      prevGopMean: 0,
      nextGopMean: 0,
      relDur: 0,
      relCharPerSec: 0,
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
  addContext(words)
  return words
}

type CharShape = Omit<
  Record<(typeof CHAR_FEATURE_KEYS)[number], number>,
  'gapBefore' | 'gapAfter' | 'prevGopMean' | 'nextGopMean' | 'relDur' | 'relCharPerSec'
>

/** Describe the shape of a word's per-character series. */
export function charFeatures(chars: GopChar[]): CharShape {
  const n = chars.length
  if (!n) {
    return {
      charGopMin: 0,
      charGopMax: 0,
      charGopStd: 0,
      charGopRange: 0,
      charGopP25: 0,
      worstCharPos: 0,
      nWeakChars: 0,
      fracWeakChars: 0,
      headGop: 0,
      tailGop: 0,
      headTailDelta: 0,
      charDurMax: 0,
      charDurMin: 0,
      charDurStd: 0,
      charDurCv: 0,
      longestCharPos: 0,
      durFrontBackRatio: 0,
      charMarginMin: 0,
      charMarginP25: 0,
    }
  }
  const gops = chars.map((c) => c.gop)
  const margins = chars.map((c) => c.margin)
  const durs = chars.map((c) => c.dur)

  const [, gMin, gStd] = stats(gops)
  const gMax = Math.max(...gops)
  let worst = 0
  for (let i = 1; i < n; i++) if (gops[i]! < gops[worst]!) worst = i
  let weak = 0
  for (const g of gops) if (g <= WEAK_CHAR_GOP) weak += 1

  const third = Math.max(1, Math.floor(n / 3))
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1)
  const head = mean(gops.slice(0, third))
  const tail = mean(gops.slice(n - third))

  const [dMean, dMin, dStd] = stats(durs)
  const dMax = Math.max(...durs)
  let longest = 0
  for (let i = 1; i < n; i++) if (durs[i]! > durs[longest]!) longest = i
  const half = Math.max(1, Math.floor(n / 2))
  const front = durs.slice(0, half).reduce((a, b) => a + b, 0) || 1e-6
  const back = durs.slice(half).reduce((a, b) => a + b, 0) || 1e-6

  const [, mMin] = stats(margins)

  // Positions are normalised so a two-character word and a ten-character word
  // say "the problem is at the end" the same way.
  const denom = Math.max(1, n - 1)
  return {
    charGopMin: gMin,
    charGopMax: gMax,
    charGopStd: gStd,
    charGopRange: gMax - gMin,
    charGopP25: quantile(
      [...gops].sort((a, b) => a - b),
      0.25,
    ),
    worstCharPos: worst / denom,
    nWeakChars: weak,
    fracWeakChars: weak / n,
    headGop: head,
    tailGop: tail,
    headTailDelta: head - tail,
    charDurMax: dMax,
    charDurMin: dMin,
    charDurStd: dStd,
    charDurCv: dMean > 1e-9 ? dStd / dMean : 0,
    longestCharPos: longest / denom,
    durFrontBackRatio: front / back,
    charMarginMin: mMin,
    charMarginP25: quantile(
      [...margins].sort((a, b) => a - b),
      0.25,
    ),
  }
}

/**
 * The v2 block for a word that has no per-character series attached.
 *
 * `wordsFromPath` always fills these. This is for the other direction —
 * building a `GopWord` from something that is not a CTC path, such as a test
 * fixture or an externally supplied timeline. A function rather than a constant
 * so no two words end up sharing one `chars` array.
 */
export function noCharSeries(): Pick<GopWord, 'chars' | (typeof CHAR_FEATURE_KEYS)[number]> {
  return {
    chars: [],
    ...charFeatures([]),
    gapBefore: 0,
    gapAfter: 0,
    prevGopMean: 0,
    nextGopMean: 0,
    relDur: 0,
    relCharPerSec: 0,
  }
}

/** Fill the neighbour-dependent features in place. */
export function addContext(words: GopWord[]): void {
  if (!words.length) return
  const meanDur = words.reduce((a, w) => a + w.dur, 0) / words.length
  const meanCps = words.reduce((a, w) => a + w.charPerSec, 0) / words.length
  for (let i = 0; i < words.length; i++) {
    const w = words[i]!
    const prev = i > 0 ? words[i - 1]! : null
    const next = i + 1 < words.length ? words[i + 1]! : null
    w.gapBefore = prev ? w.t0 - prev.t1 : 0
    w.gapAfter = next ? next.t0 - w.t1 : 0
    w.prevGopMean = prev ? prev.gopMean : 0
    w.nextGopMean = next ? next.gopMean : 0
    w.relDur = meanDur > 1e-9 ? w.dur / meanDur : 0
    w.relCharPerSec = meanCps > 1e-9 ? w.charPerSec / meanCps : 0
  }
}

/** Feature vector for a word-level head, in `WORD_FEATURE_KEYS` order. */
export function wordVector(w: GopWord): number[] {
  return WORD_FEATURE_KEYS.map((k) => {
    const v = w[k]
    return typeof v === 'number' && Number.isFinite(v) ? v : 0
  })
}

/** Feature vector for a v2 word-level head, in `WORD_FEATURE_KEYS_V2` order. */
export function wordVectorV2(w: GopWord): number[] {
  return wordVectorBy(w, WORD_FEATURE_KEYS_V2)
}

/**
 * Feature vector in a caller-supplied key order.
 *
 * A shipped head carries the exact key list it was fitted on, so the runtime can
 * assemble its input from that list rather than from a version number. One head
 * generation more and the serving code stays as it is; a head asking for a key
 * this build does not produce fails loudly here instead of being fed a zero.
 */
export function wordVectorBy(w: GopWord, keys: readonly string[]): number[] {
  return keys.map((k) => {
    if (!(k in w)) throw new Error(`scoring head wants unknown word feature ${k}`)
    const v = w[k as keyof GopWord]
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

export function utteranceGopFeaturesV2(words: GopWord[]): UtteranceGopFeaturesV2 {
  const base = utteranceGopFeatures(words)
  if (!words.length) {
    const empty = {} as UtteranceGopFeaturesV2
    for (const k of UTTERANCE_GOP_KEYS_V2) empty[k] = 0
    return empty
  }
  const col = (pick: (w: GopWord) => number) => words.map(pick)
  const charMin = col((w) => w.charGopMin)
  const weak = col((w) => w.fracWeakChars)
  const delta = col((w) => w.headTailDelta)
  const cv = col((w) => w.charDurCv)
  const gaps = col((w) => w.gapBefore)
  const g = col((w) => w.gopMean)
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

  return {
    ...base,
    charGopMinMean: mean(charMin),
    charGopMinP10: quantile(
      [...charMin].sort((a, b) => a - b),
      0.1,
    ),
    charGopStdMean: mean(col((w) => w.charGopStd)),
    fracWeakCharsMean: mean(weak),
    fracWordsWithWeakChar: weak.filter((x) => x > 0).length / weak.length,
    headTailDeltaMean: mean(delta),
    headTailDeltaStd: stats(delta)[2],
    charDurCvMean: mean(cv),
    charDurCvStd: stats(cv)[2],
    worstCharPosMean: mean(col((w) => w.worstCharPos)),
    longestCharPosMean: mean(col((w) => w.longestCharPos)),
    relDurStd: stats(col((w) => w.relDur))[2],
    gapBeforeMax: Math.max(...gaps),
    gapBeforeMean: mean(gaps),
    wordGopRange: Math.max(...g) - Math.min(...g),
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
