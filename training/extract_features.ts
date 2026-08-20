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
import { appendFileSync, createReadStream, existsSync, readFileSync, statSync } from 'node:fs'
import { appendFile, mkdir, truncate, writeFile } from 'node:fs/promises'
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
/** wav2vec2-base. The sidecar is raw float32, so its stride has to be known to read it back. */
const HIDDEN_SIZE = 768
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
    // Pooled hidden layers go to a sidecar rather than into the JSONL: they are
    // hundreds of float32 per word, and a JSON number per component would cost
    // an order of magnitude more to write and to parse than the raw bytes.
    hidden: get('--hidden') ? resolve(get('--hidden')!) : null,
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

async function doneIds(path: string): Promise<{ ids: Set<string>; hiddenRows: number }> {
  const ids = new Set<string>()
  let hiddenRows = 0
  if (!existsSync(path)) return { ids, hiddenRows }
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity })
  for await (const line of rl) {
    if (!line.trim()) continue
    try {
      const row = JSON.parse(line) as { utteranceId: string; words: unknown[] }
      ids.add(row.utteranceId)
      hiddenRows += row.words.length + 1
    } catch {
      // A run killed mid-write leaves one torn line; everything before it stands.
    }
  }
  return { ids, hiddenRows }
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
  const todo = rows.filter((r) => !already.ids.has(r.utteranceId))
  console.log(`${rows.length} utterances, ${already.ids.size} already extracted, ${todo.length} to go`)

  await mkdir(dirname(args.out), { recursive: true })
  if (args.hidden) {
    // Repair a sidecar left long by a kill between the two writes. Anything
    // shorter than the JSONL implies means the two files came from different
    // runs, and silently continuing would misalign every later word against
    // its label -- a fault that shows up as a slightly worse model and
    // nothing else.
    const size = HIDDEN_SIZE
    const want = already.hiddenRows * size * 4
    const have = existsSync(args.hidden) ? statSync(args.hidden).size : 0
    if (have > want) {
      await truncate(args.hidden, want)
      console.log(`sidecar truncated ${have} -> ${want} bytes to match the JSONL`)
    } else if (have < want) {
      console.error(
        `${args.hidden} holds ${have} bytes, the JSONL implies ${want}. ` +
          'Delete both and re-extract.',
      )
      process.exit(1)
    }
  }
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
      result = await analyzer.analyze({
        audio,
        referenceText: row.text,
        includeHidden: args.hidden !== null,
      })
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
        // The aligned frame span. Nothing in the shipped feature set needs it,
        // but an experiment that wants to summarise this word's audio some
        // other way has to know which frames are this word's, and recomputing
        // the alignment separately would not be the same alignment.
        t0: gopWords[i]!.t0,
        t1: gopWords[i]!.t1,
        accuracy: w.accuracy,
        stress: w.stress,
        total: w.total,
        level: w.level,
      })),
    }
    if (args.hidden) {
      if (!result.hidden) throw new Error('asked for hidden layers, acoustic model emits none')
      // One row per word then one for the clip, float32, in JSONL order.
      // Written *before* the JSONL line on purpose: a run killed between the
      // two writes can then only leave the sidecar long, which the startup
      // truncation below repairs exactly. The other order would leave it short,
      // and nothing but re-extracting could tell which utterance was missing.
      const size = result.hidden.size
      const buf = new Float32Array((aligned.length + 1) * size)
      result.hidden.words.forEach((w, i) => buf.set(w, i * size))
      buf.set(result.hidden.utterance, aligned.length * size)
      appendFileSync(args.hidden, Buffer.from(buf.buffer, buf.byteOffset, buf.byteLength))
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
