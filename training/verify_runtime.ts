#!/usr/bin/env npx tsx
/**
 * Check that serving reproduces evaluation.
 *
 *   npx tsx training/verify_runtime.ts [--limit N]
 *
 * eval.py measures the ONNX graphs in Python against extracted features.
 * This runs the whole Node path — decode, align, GOP, prosody, scoring — over
 * corpus audio and compares the word bands it produces against the same labels.
 * If the two disagree, something between feature extraction and serving moved,
 * which is the failure the shared extraction path exists to prevent.
 */
import { createReadStream, existsSync, readFileSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { ReadAloudAnalyzer } from '@readaloudkit/core'
import { scoringReady } from '@readaloudkit/scoring'

const HERE = dirname(fileURLToPath(import.meta.url))
const CORPUS = resolve(HERE, 'data', 'speechocean762')
const LEVELS = ['good', 'average', 'bad'] as const

type Row = {
  split: string
  text: string
  audioPath: string
  words: { level: string }[]
}

/**
 * Sample evenly across the split rather than taking the head of it. Utterance
 * ids are grouped by speaker, so the first N rows are a handful of speakers,
 * and speaker-to-speaker variance here is large — the first 250 utterances
 * score 0.572 where the whole split scores 0.726.
 */
async function readRows(path: string, limit: number): Promise<Row[]> {
  const all: Row[] = []
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity })
  for await (const line of rl) {
    if (!line.trim()) continue
    const row = JSON.parse(line) as Row
    if (row.split === 'test') all.push(row)
  }
  if (!limit || limit >= all.length) return all
  const stride = all.length / limit
  return Array.from({ length: limit }, (_, i) => all[Math.floor(i * stride)]!)
}

async function main() {
  const argv = process.argv.slice(2).filter((a) => a !== '--')
  const li = argv.indexOf('--limit')
  const limit = li >= 0 ? Number(argv[li + 1]) : 250

  const utterances = resolve(HERE, 'data', 'utterances.jsonl')
  if (!existsSync(utterances)) {
    console.error('run training/prepare.py first')
    process.exit(1)
  }
  if (!scoringReady()) {
    console.error('no scoring heads installed under models/scoring')
    process.exit(1)
  }

  const rows = await readRows(utterances, limit)
  const analyzer = new ReadAloudAnalyzer()
  const backend = await analyzer.scoringBackend()
  console.log(`scoring backend: ${backend.backend} ${backend.version ?? '(unversioned)'}`)

  const tally: Record<string, Record<string, number>> = {}
  let n = 0
  let agree = 0
  let flagged = 0
  let flaggedReal = 0
  let real = 0
  let mismatched = 0
  let unbanded = 0
  let banded = 0

  for (const row of rows) {
    const audio = new Uint8Array(readFileSync(resolve(CORPUS, row.audioPath)))
    const result = await analyzer.analyze({ audio, referenceText: row.text })
    const aligned = result.words.filter((w) => w.reference !== null && w.gop !== null)
    if (aligned.length !== row.words.length) {
      mismatched += 1
      continue
    }
    aligned.forEach((w, i) => {
      const truth = row.words[i]!.level
      n += 1
      const isReal = truth !== 'good'
      if (isReal) real += 1
      if (w.needsAttention) {
        flagged += 1
        if (isReal) flaggedReal += 1
      }
      // The runtime withholds a band from words it reads as unspoken. They are
      // not head predictions, so they belong outside the confusion rather than
      // silently in a column nobody prints.
      if (w.level === null) {
        unbanded += 1
        return
      }
      const got = String(w.level)
      tally[truth] ??= {}
      tally[truth][got] = (tally[truth][got] ?? 0) + 1
      banded += 1
      if (truth === got) agree += 1
    })
  }

  console.log(`\n${n} words from ${rows.length} test utterances (${mismatched} skipped)`)
  console.log(`${unbanded} left unbanded as omissions; agreement is over the remaining ${banded}`)
  console.log(`level agreement with the corpus label: ${((100 * agree) / banded).toFixed(1)}%`)
  console.log('confusion (rows = corpus label, columns = predicted):')
  for (const t of LEVELS) {
    const row = LEVELS.map((g) => `${g}:${tally[t]?.[g] ?? 0}`).join('  ')
    console.log(`  ${t.padEnd(8)} ${row}`)
  }
  const precision = flagged ? flaggedReal / flagged : NaN
  console.log(
    `\nneedsAttention: flagged ${flagged} of ${n}` +
      `  precision ${precision.toFixed(3)}  recall ${(flaggedReal / real).toFixed(3)}` +
      `  (${real} words really need attention)`,
  )
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
