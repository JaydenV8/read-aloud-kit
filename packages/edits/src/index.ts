export const MIN_OMIT_RUN = 3
export const MIN_EDGE_OMIT = 2
export const MIN_INSERT_RUN = 2

export const FUNCTION_WORDS = new Set([
  'a',
  'an',
  'the',
  'to',
  'of',
  'in',
  'on',
  'at',
  'for',
  'from',
  'by',
  'as',
  'and',
  'or',
  'but',
  'is',
  'are',
  'was',
  'were',
  'be',
  'been',
  'am',
  'do',
  'does',
  'did',
  'that',
  'this',
  'these',
  'those',
  'who',
  'whom',
  'which',
  'you',
  'it',
  'he',
  'she',
  'they',
  'we',
  'them',
  'his',
  'her',
  'their',
  'with',
  'if',
  'so',
  'than',
  'then',
  'not',
  'can',
  'will',
  'would',
  'could',
  'should',
  'have',
  'has',
  'had',
  'into',
  'about',
  'over',
  'after',
  'before',
  'up',
  'out',
  'off',
])

const WORD_RE = /[A-Za-z']+/g
const VOWELS = /[aeiou]/g

export type EditOp = {
  op: 'equal' | 'replace' | 'omit' | 'insert' | 'repeat' | 'skip'
  ref: string | null
  hyp: string | null
  soft?: string
}

export function tokenize(text: string | Iterable<string> | null | undefined): string[] {
  if (text == null) return []
  if (typeof text === 'string') return (text.match(WORD_RE) ?? []).map((w) => w.toLowerCase())
  const out: string[] = []
  for (const item of text) {
    if (!item) continue
    out.push(...(String(item).match(WORD_RE) ?? []).map((w) => w.toLowerCase()))
  }
  return out
}

export function wordEdits(ref: string | string[], hyp: string | string[]): EditOp[] {
  const r = tokenize(ref)
  const h = tokenize(hyp)
  const n = r.length
  const m = h.length
  const dp: number[][] = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0))
  const bt: string[][] = Array.from({ length: n + 1 }, () => Array(m + 1).fill(''))
  for (let i = 1; i <= n; i++) {
    dp[i]![0] = i
    bt[i]![0] = 'omit'
  }
  for (let j = 1; j <= m; j++) {
    dp[0]![j] = j
    bt[0]![j] = 'insert'
  }
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (r[i - 1] === h[j - 1]) {
        dp[i]![j] = dp[i - 1]![j - 1]!
        bt[i]![j] = 'equal'
      } else {
        const cand: [number, string][] = [
          [dp[i - 1]![j - 1]! + 1, 'replace'],
          [dp[i - 1]![j]! + 1, 'omit'],
          [dp[i]![j - 1]! + 1, 'insert'],
        ]
        cand.sort((a, b) => a[0] - b[0])
        dp[i]![j] = cand[0]![0]
        bt[i]![j] = cand[0]![1]
      }
    }
  }
  const ops: EditOp[] = []
  let i = n
  let j = m
  while (i > 0 || j > 0) {
    const op = bt[i]![j] || ''
    if (op === 'equal' || op === 'replace') {
      ops.push({ op, ref: r[i - 1]!, hyp: h[j - 1]! })
      i -= 1
      j -= 1
    } else if (op === 'omit') {
      ops.push({ op: 'omit', ref: r[i - 1]!, hyp: null })
      i -= 1
    } else if (op === 'insert') {
      ops.push({ op: 'insert', ref: null, hyp: h[j - 1]! })
      j -= 1
    } else {
      break
    }
  }
  ops.reverse()
  for (let k = 0; k < ops.length; k++) {
    const rec = ops[k]!
    if (rec.op !== 'insert' || !rec.hyp) continue
    const neighbors: string[] = []
    if (k > 0) {
      const prev = ops[k - 1]!
      neighbors.push(prev.hyp || prev.ref || '')
    }
    if (k + 1 < ops.length) {
      const nxt = ops[k + 1]!
      neighbors.push(nxt.hyp || nxt.ref || '')
    }
    if (neighbors.includes(rec.hyp)) rec.op = 'repeat'
  }
  return ops
}

function levenshtein(a: string, b: string): number {
  if (a === b) return 0
  if (!a) return b.length
  if (!b) return a.length
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j)
  for (let i = 1; i <= a.length; i++) {
    const cur = [i]
    for (let j = 1; j <= b.length; j++) {
      cur.push(
        Math.min(prev[j]! + 1, cur[j - 1]! + 1, prev[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1)),
      )
    }
    prev = cur
  }
  return prev[b.length]!
}

export function nearPhone(a: string, b: string): boolean {
  a = (a || '').toLowerCase()
  b = (b || '').toLowerCase()
  if (!a || !b || a === b) return true
  if (FUNCTION_WORDS.has(a) && FUNCTION_WORDS.has(b)) return true
  const d = levenshtein(a, b)
  const longer = Math.max(a.length, b.length)
  if (d <= 1 && longer >= 3) return true
  if (d <= 2 && longer >= 6) return true
  const ca = a.replace(VOWELS, '')
  const cb = b.replace(VOWELS, '')
  if (ca && ca === cb && Math.min(a.length, b.length) >= 4) return true
  if (ca && cb && Math.min(ca.length, cb.length) >= 3 && levenshtein(ca, cb) <= 1) return true
  return false
}

function omitAtEdge(ops: EditOp[], start: number, end: number): boolean {
  const prefix = ops.slice(0, start).every((o) => ['insert', 'repeat', 'skip'].includes(o.op))
  const suffix = ops.slice(end).every((o) => ['insert', 'repeat', 'skip'].includes(o.op))
  return prefix || suffix
}

export function softenOps(ops: EditOp[]): EditOp[] {
  const out = ops.map((rec) => ({ ...rec }))
  for (const rec of out) {
    if (rec.op !== 'replace') continue
    const ref = rec.ref || ''
    const hyp = rec.hyp || ''
    if (FUNCTION_WORDS.has(ref) || FUNCTION_WORDS.has(hyp) || nearPhone(ref, hyp)) {
      rec.op = 'equal'
      rec.soft = 'near'
    }
  }
  const n = out.length
  const chargeOmit = Array(n).fill(false)
  let i = 0
  while (i < n) {
    if (out[i]!.op !== 'omit') {
      i += 1
      continue
    }
    let j = i + 1
    while (j < n && out[j]!.op === 'omit') j += 1
    const thresh = omitAtEdge(out, i, j) ? MIN_EDGE_OMIT : MIN_OMIT_RUN
    if (j - i >= thresh) {
      for (let k = i; k < j; k++) chargeOmit[k] = true
    }
    i = j
  }
  for (let k = 0; k < n; k++) {
    const rec = out[k]!
    if (rec.op === 'omit' && !chargeOmit[k]) {
      rec.op = 'equal'
      rec.soft = 'short_omit'
      rec.hyp = rec.ref
    }
  }
  i = 0
  while (i < n) {
    if (out[i]!.op !== 'insert') {
      i += 1
      continue
    }
    let j = i + 1
    while (j < n && out[j]!.op === 'insert') j += 1
    const run = j - i
    for (let k = i; k < j; k++) {
      const hyp = out[k]!.hyp || ''
      if (FUNCTION_WORDS.has(hyp) || run < MIN_INSERT_RUN) {
        out[k]!.op = 'skip'
        out[k]!.soft = 'short_insert'
      }
    }
    i = j
  }
  return out
}

export function contentFromEdits(
  nRef: number,
  nOmission: number,
  nReplacement = 0,
  nInsertion = 0,
): number {
  if (nRef <= 0) return 0
  const errors = Math.max(0, nOmission + nReplacement + nInsertion)
  return Math.max(0, Math.min(5, (5 * (nRef - errors)) / nRef))
}

export function countEdits(ops: EditOp[]) {
  const nOmit = ops.filter((o) => o.op === 'omit').length
  const nRep = ops.filter((o) => o.op === 'replace').length
  const nIns = ops.filter((o) => o.op === 'insert').length
  const nRepeat = ops.filter((o) => o.op === 'repeat').length
  const nEq = ops.filter((o) => o.op === 'equal').length
  const nRef = nEq + nOmit + nRep
  return {
    nRef,
    nEqual: nEq,
    nOmission: nOmit,
    nReplacement: nRep,
    nInsertion: nIns,
    nRepeat,
    nContentErrors: nOmit + nRep + nIns,
  }
}

export function editsFromTexts(refText: string, hypText: string, calibrate = true) {
  let ops = wordEdits(refText, hypText)
  if (calibrate) ops = softenOps(ops)
  const counts = countEdits(ops)
  return {
    ...counts,
    content: contentFromEdits(
      counts.nRef,
      counts.nOmission,
      counts.nReplacement,
      counts.nInsertion,
    ),
    ops,
    calibrated: calibrate,
  }
}

export type DualTokenKind = 'word' | 'omission' | 'replacement' | 'insert' | 'repeat'

/**
 * `align` is passed through untouched from `alignWords`, so the caller keeps
 * whatever per-word evidence it started with. Narrowing it here would silently
 * drop acoustic fields the scoring heads need.
 */
export type DualToken<A = unknown> = {
  tok: string
  display: string
  kind: DualTokenKind
  hyp: string | null
  align: A | null
}

export function dualTrackTokens<A>(
  alignWords: A[],
  refText: string,
  hyp: string | string[],
): { tokens: DualToken<A>[]; counts: ReturnType<typeof editsFromTexts> } {
  const ops = softenOps(wordEdits(refText, hyp))
  const counts = countEdits(ops)
  const packed = {
    ...counts,
    content: contentFromEdits(
      counts.nRef,
      counts.nOmission,
      counts.nReplacement,
      counts.nInsertion,
    ),
    ops,
    calibrated: true as const,
  }
  const tokens: DualToken<A>[] = []
  let ai = 0
  for (const op of ops) {
    if (op.op === 'skip') continue
    if (op.op === 'insert' || op.op === 'repeat') {
      tokens.push({
        tok: op.hyp || '',
        display: op.hyp || '',
        kind: op.op,
        hyp: op.hyp,
        align: null,
      })
      continue
    }
    const aw = ai < alignWords.length ? alignWords[ai]! : null
    ai += 1
    const kind = { equal: 'word', replace: 'replacement', omit: 'omission' }[op.op] as DualTokenKind
    tokens.push({
      tok: op.ref || '',
      display: op.ref || '',
      kind,
      hyp: kind === 'replacement' ? op.hyp : null,
      align: aw,
    })
  }
  return { tokens, counts: packed }
}
