export const WORD_SEP = '|'

export type CtcLabels = {
  blank: number
  labels: string[]
  sampleRate: number
  wordSep?: string
}

export function collapseIds(ids: number[], blank = 0): number[] {
  const out: number[] = []
  let prev: number | null = null
  for (const raw of ids) {
    const i = raw | 0
    if (i === blank) {
      prev = blank
      continue
    }
    if (prev === i) continue
    out.push(i)
    prev = i
  }
  return out
}

export function idsToText(ids: number[], labels: string[], blank = 0): string {
  const chars: string[] = []
  for (const i of collapseIds(ids, blank)) {
    if (i >= 0 && i < labels.length) chars.push(labels[i]!)
  }
  return chars.join('')
}

export function idsToWords(ids: number[], labels: string[], blank = 0): string[] {
  return idsToText(ids, labels, blank)
    .split(WORD_SEP)
    .map((w) => w.toLowerCase())
    .filter(Boolean)
}

export function greedyIds(logits: Float32Array, frames: number, vocabSize: number): number[] {
  const ids = Array.from({ length: frames }, () => 0)
  for (let t = 0; t < frames; t++) {
    let best = 0
    let bestV = -Infinity
    const off = t * vocabSize
    for (let c = 0; c < vocabSize; c++) {
      const v = logits[off + c]!
      if (v > bestV) {
        bestV = v
        best = c
      }
    }
    ids[t] = best
  }
  return ids
}

export function logSoftmaxRows(
  logits: Float32Array,
  frames: number,
  vocabSize: number,
): Float32Array {
  const out = new Float32Array(logits.length)
  for (let t = 0; t < frames; t++) {
    const off = t * vocabSize
    let max = -Infinity
    for (let c = 0; c < vocabSize; c++) max = Math.max(max, logits[off + c]!)
    let sum = 0
    for (let c = 0; c < vocabSize; c++) {
      const e = Math.exp(logits[off + c]! - max)
      out[off + c] = e
      sum += e
    }
    const logSum = Math.log(sum)
    for (let c = 0; c < vocabSize; c++) {
      out[off + c] = logits[off + c]! - max - logSum
    }
  }
  return out
}

export function tokenizeReference(text: string, dictionary: Record<string, number>): number[] {
  const words = (text.match(/[A-Za-z']+/g) ?? []) as string[]
  const pieces: string[] = []
  for (let i = 0; i < words.length; i++) {
    if (i) pieces.push('|')
    pieces.push(...words[i]!.toUpperCase().split(''))
  }
  const ids: number[] = []
  for (const ch of pieces) {
    const id = dictionary[ch]
    if (id !== undefined) ids.push(id)
  }
  return ids
}

export function labelIndex(labels: string[]): Record<string, number> {
  const d: Record<string, number> = {}
  for (let i = 0; i < labels.length; i++) d[labels[i]!] = i
  return d
}
