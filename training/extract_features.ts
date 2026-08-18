#!/usr/bin/env npx tsx
/**
 * Turn prepared utterances into model inputs.
 *
 * This deliberately calls the same `ReadAloudAnalyzer.analyze` the HTTP API
 * calls, rather than reassembling the pipeline for training. A separate
 * extraction path is the classic way to ship a model that scores differently in
 * production than it did in evaluation, and the failure is silent.
 *
 *   npx tsx training/extract_features.ts [--limit N] [--adults-only]
 *
 * Interrupting is safe: finished utterances are skipped on the next run.
 */
import { createReadStream, existsSync, readFileSync } from 'node:fs'
import { appendFile, mkdir, writeFile } from 'node:fs/promises'
import { createInterface } from 'node:readline'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { ReadAloudAnalyzer } from '@readaloudkit/core'
import { UTTERANCE_FEATURE_KEYS, utteranceVector } from '@readaloudkit/features'
import {
  UTTERANCE_GOP_KEYS,
  UTTERANCE_GOP_KEYS_V2,
  WEAK_CHAR_GOP,
  WORD_FEATURE_KEYS,
  WORD_FEATURE_KEYS_V2,
  utteranceGopFeatures,
  utteranceGopFeaturesV2,
  wordVector,
  wordVectorV2,
} from '@readaloudkit/gop'

const HERE = dirname(fileURLToPath(import.meta.url))
const CORPUS = resolve(HERE, 'data', 'speechocean762')

type PreparedWord = {
  text: string
  accuracy: number
  stress: number
  total: number
  level: string
}

type Prepared = {
  utteranceId: string
  speakerId: string
  split: string
  text: string
  audioPath: string
  durationSec: number
  speaker: { age: number; gender: string; adult: boolean }
  labels: Record<string, number>
  words: PreparedWord[]
}

function parseArgs() {
  const argv = process.argv.slice(2).filter((a) => a !== '--')
  const get = (name: string) => {
    const i = argv.indexOf(name)
    return i >= 0 ? argv[i + 1] : undefined
  }
  return {
    input: resolve(get('--input') ?? resolve(HERE, 'data', 'utterances.jsonl')),
    out: resolve(get('--out') ?? resolve(HERE, 'data', 'features.jsonl')),
    limit: Number(get('--limit') ?? 0) || 0,
    adultsOnly: argv.includes('--adults-only'),
  }
}

async function readJsonl<T>(path: string): Promise<T[]> {
  const rows: T[] = []
  if (!existsSync(path)) return rows
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity })
  for await (const line of rl) {
    if (line.trim()) rows.push(JSON.parse(line) as T)
  }
  return rows
}

async function doneIds(path: string): Promise<Set<string>> {
  const ids = new Set<string>()
  if (!existsSync(path)) return ids
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity })
  for await (const line of rl) {
    if (!line.trim()) continue
    try {
      ids.add((JSON.parse(line) as { utteranceId: string }).utteranceId)
    } catch {
      // A run killed mid-write leaves one torn line; everything before it stands.
    }
  }
  return ids
}

async function main() {
  const args = parseArgs()
  const analyzer = new ReadAloudAnalyzer()
  if (!analyzer.ready()) {
    console.error('acoustic model missing — run `pnpm models:download` first')
    process.exit(1)
  }

  let rows = await readJsonl<Prepared>(args.input)
  if (args.adultsOnly) rows = rows.filter((r) => r.speaker.adult)
  if (args.limit) rows = rows.slice(0, args.limit)

  const already = await doneIds(args.out)
  const todo = rows.filter((r) => !already.has(r.utteranceId))
  console.log(`${rows.length} utterances, ${already.size} already extracted, ${todo.length} to go`)

  await mkdir(dirname(args.out), { recursive: true })
  await writeFile(
    resolve(dirname(args.out), 'features.meta.json'),
    JSON.stringify(
      {
        wordFeatureKeys: WORD_FEATURE_KEYS,
        utteranceFeatureKeys: [...UTTERANCE_FEATURE_KEYS, ...UTTERANCE_GOP_KEYS],
        wordFeatureKeysV2: WORD_FEATURE_KEYS_V2,
        utteranceFeatureKeysV2: [...UTTERANCE_FEATURE_KEYS, ...UTTERANCE_GOP_KEYS_V2],
        weakCharGop: WEAK_CHAR_GOP,
        note: 'Feature order is the model input contract. Regenerate features if it changes.',
      },
      null,
      2,
    ) + '\n',
  )

  const started = Date.now()
  let done = 0
  let skipped = 0
  for (const row of todo) {
    const audio = new Uint8Array(readFileSync(resolve(CORPUS, row.audioPath)))
    let result
    try {
      result = await analyzer.analyze({ audio, referenceText: row.text })
    } catch (e) {
      console.warn(`  ${row.utteranceId}: ${(e as Error).message}`)
      skipped += 1
      continue
    }

    // One aligned span per reference word; insertions carry no alignment.
    const aligned = result.words.filter((w) => w.reference !== null && w.gop !== null)
    if (aligned.length !== row.words.length) {
      console.warn(
        `  ${row.utteranceId}: ${aligned.length} aligned vs ${row.words.length} labelled, skipping`,
      )
      skipped += 1
      continue
    }

    const gopWords = aligned.map((w) => w.gop!)
    const uttGop = utteranceGopFeatures(gopWords)
    const uttGopV2 = utteranceGopFeaturesV2(gopWords)
    const prosody = utteranceVector(result.prosody)
    const record = {
      utteranceId: row.utteranceId,
      speakerId: row.speakerId,
      split: row.split,
      adult: row.speaker.adult,
      durationSec: row.durationSec,
      hypothesis: result.hypothesis,
      utterance: {
        features: [...prosody, ...UTTERANCE_GOP_KEYS.map((k) => uttGop[k])],
        featuresV2: [...prosody, ...UTTERANCE_GOP_KEYS_V2.map((k) => uttGopV2[k])],
        labels: row.labels,
      },
      words: row.words.map((w, i) => ({
        text: w.text,
        features: wordVector(gopWords[i]!),
        featuresV2: wordVectorV2(gopWords[i]!),
        // The raw series, so the weak-character threshold can be swept on this
        // corpus rather than inherited from somewhere it was not measured.
        charGops: gopWords[i]!.chars.map((c) => Math.round(c.gop * 1e6) / 1e6),
        accuracy: w.accuracy,
        stress: w.stress,
        total: w.total,
        level: w.level,
      })),
    }
    await appendFile(args.out, JSON.stringify(record) + '\n')

    done += 1
    if (done % 100 === 0 || done === todo.length) {
      const rate = done / ((Date.now() - started) / 1000)
      const left = Math.round((todo.length - done) / Math.max(rate, 1e-6))
      console.log(
        `  ${done}/${todo.length}  ${rate.toFixed(1)}/s  ~${Math.round(left / 60)} min left`,
      )
    }
  }

  console.log(`extracted ${done}, skipped ${skipped} -> ${args.out}`)
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
