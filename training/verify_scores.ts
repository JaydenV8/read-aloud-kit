#!/usr/bin/env npx tsx
/**
 * Check that the utterance scores a request returns reproduce evaluation.
 *
 *   npx tsx training/verify_scores.ts [--limit N]
 *
 * `verify_runtime.ts` checks word bands. The utterance heads are where 0.6
 * claims most of its gain, and they are assembled from a different vector by a
 * different code path, so the claim has to be checked where a caller would see
 * it: through `analyze`, on corpus audio, against the same expert labels
 * `eval.py` uses.
 */
import { createReadStream, existsSync, readFileSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { ReadAloudAnalyzer } from '@readaloudkit/core'

const HERE = dirname(fileURLToPath(import.meta.url))
const CORPUS = resolve(HERE, 'data', 'speechocean762')

type Row = {
  utteranceId: string
  split: string
  text: string
  audioPath: string
  labels: Record<string, number>
}

function pearson(a: number[], b: number[]): number {
  const n = a.length
  const ma = a.reduce((x, y) => x + y, 0) / n
  const mb = b.reduce((x, y) => x + y, 0) / n
  let num = 0
  let da = 0
  let db = 0
  for (let i = 0; i < n; i++) {
    const x = a[i]! - ma
    const y = b[i]! - mb
    num += x * y
    da += x * x
    db += y * y
  }
  return num / Math.sqrt(da * db)
}

async function main() {
  const argv = process.argv.slice(2)
  const limit = Number(argv[argv.indexOf('--limit') + 1] ?? 0) || 300
  const path = resolve(HERE, 'data', 'utterances.jsonl')
  const all: Row[] = []
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity })
  for await (const line of rl) {
    if (!line.trim()) continue
    const r = JSON.parse(line) as Row
    if (r.split === 'test') all.push(r)
  }
  // Even stride, not the head of the file: ids are grouped by speaker.
  const step = Math.max(1, Math.floor(all.length / limit))
  const rows = all.filter((_, i) => i % step === 0).slice(0, limit)

  const analyzer = new ReadAloudAnalyzer()
  const backend = await analyzer.scoringBackend()
  console.log(`scoring backend: ${backend.backend} ${backend.version ?? '(unversioned)'}`)

  // The heads are fitted on the corpus 0-10 scale; the runtime reports 10-90.
  // Correlation is invariant to that, so the labels need no conversion.
  const got: Record<string, number[]> = { pronunciation: [], fluency: [], overall: [] }
  const want: Record<string, number[]> = { pronunciation: [], fluency: [], overall: [] }
  const field = { pronunciation: 'accuracy', fluency: 'fluency', overall: 'total' } as const
  let skipped = 0
  for (const r of rows) {
    const file = resolve(CORPUS, r.audioPath)
    if (!existsSync(file)) { skipped++; continue }
    const out = await analyzer.analyze({
      audio: new Uint8Array(readFileSync(file)),
      referenceText: r.text,
    })
    for (const k of ['pronunciation', 'fluency', 'overall'] as const) {
      const v = out.scores[k]
      if (v === null) { continue }
      got[k]!.push(v)
      want[k]!.push(r.labels[field[k]]!)
    }
  }
  console.log(`${rows.length - skipped} test utterances\n`)
  for (const k of ['pronunciation', 'fluency', 'overall'] as const) {
    console.log(`  ${k.padEnd(14)} r=${pearson(got[k]!, want[k]!).toFixed(4)}  n=${got[k]!.length}`)
  }
}

main()
