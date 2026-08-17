import type { AlignmentSpan } from '@readaloudkit/types'

const NEG = -1e30

/**
 * CTC Viterbi forced alignment (same lattice as torchaudio.functional.forced_align).
 * logProbs is row-major [T, C].
 */
export function alignReference(input: {
  logProbs: Float32Array
  frames: number
  vocabSize: number
  referenceTokens: number[]
  blankId: number
}): { path: number[]; spans: AlignmentSpan[] } {
  const { logProbs, frames: T, vocabSize: C, referenceTokens, blankId } = input
  const L = referenceTokens.length
  if (T === 0 || L === 0) return { path: Array.from({ length: T }, () => blankId), spans: [] }

  const tokens: number[] = [blankId]
  for (const t of referenceTokens) {
    tokens.push(t)
    tokens.push(blankId)
  }
  const S = tokens.length

  const dp = new Float64Array(T * S)
  const bt = new Int16Array(T * S)
  dp.fill(NEG)
  bt.fill(-1)

  const emit = (t: number, tok: number) => logProbs[t * C + tok]!

  dp[0] = emit(0, tokens[0]!)
  if (S > 1) dp[1] = emit(0, tokens[1]!)

  for (let t = 1; t < T; t++) {
    for (let s = 0; s < S; s++) {
      const tok = tokens[s]!
      let best = dp[(t - 1) * S + s]!
      let from = s
      if (s - 1 >= 0 && dp[(t - 1) * S + (s - 1)]! > best) {
        best = dp[(t - 1) * S + (s - 1)]!
        from = s - 1
      }
      if (
        s - 2 >= 0 &&
        tok !== blankId &&
        tok !== tokens[s - 2] &&
        dp[(t - 1) * S + (s - 2)]! > best
      ) {
        best = dp[(t - 1) * S + (s - 2)]!
        from = s - 2
      }
      if (best <= NEG / 2) continue
      dp[t * S + s] = best + emit(t, tok)
      bt[t * S + s] = from
    }
  }

  let last = S - 1
  if (S > 1 && dp[(T - 1) * S + (S - 2)]! > dp[(T - 1) * S + last]!) last = S - 2

  const states = Array.from({ length: T }, () => 0)
  let s = last
  for (let t = T - 1; t >= 0; t--) {
    states[t] = s
    s = bt[t * S + s]!
    if (s < 0) s = 0
  }

  const path = states.map((st) => tokens[st]!)
  const spans: AlignmentSpan[] = []
  let i = 0
  while (i < T) {
    const id = path[i]!
    let j = i + 1
    while (j < T && path[j] === id) j += 1
    let score = 0
    for (let t = i; t < j; t++) score += emit(t, id)
    spans.push({ tokenId: id, startFrame: i, endFrame: j - 1, score: score / (j - i) })
    i = j
  }
  return { path, spans }
}
