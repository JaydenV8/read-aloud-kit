import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { UTTERANCE_FEATURE_KEYS, utteranceVector } from '@readaloudkit/features'
import { UTTERANCE_GOP_KEYS, utteranceGopFeatures, wordVector } from '@readaloudkit/gop'
import type {
  ScoredWord,
  ScoringBackend,
  ScoringInput,
  ScoringResult,
  WordLevel,
} from '@readaloudkit/types'

/**
 * Probability of `average` or `bad` above which a word is worth pointing at.
 *
 * Chosen from the test split rather than picked: at 0.8 the collapsed decision
 * runs precision 0.42 / recall 0.56 and flags about one word in six, against a
 * true rate of one in eight. Loosening to 0.5 flags better than a third of every
 * utterance at precision 0.27, and tightening to 0.9 finds a tenth of the real
 * problems. See MODEL_CARD.md.
 */
export const ATTENTION_THRESHOLD = 0.8

export type ScoringManifest = {
  /** Which release these weights are, e.g. `0.4-community`. */
  version: string
  inputName: string
  wordFeatureKeys: string[]
  utteranceFeatureKeys: string[]
  wordLevels: WordLevel[]
  displayScale: { from: [number, number]; to: [number, number] }
  heads: Record<string, { file: string; shipped: boolean }>
  outputs: Record<string, string>
}

export class NoopScoringBackend implements ScoringBackend {
  readonly name = 'none'
  async score(_input: ScoringInput): Promise<ScoringResult | null> {
    return null
  }
}

/**
 * Roots that might hold a `releases/` tree, nearest first.
 *
 * The package is consumed as raw TypeScript, so `import.meta.url` points into
 * the checkout wherever it is installed from.
 */
function repoRoots(): string[] {
  const here = dirname(fileURLToPath(import.meta.url))
  return [process.cwd(), resolve(process.cwd(), '../..'), resolve(here, '../../..')]
}

/**
 * The release named in `releases/CURRENT`, if that directory is present.
 *
 * The pointer file is what keeps the version out of the code: shipping a new
 * generation is a new directory plus one line, not an edit here.
 */
function shippedReleaseDir(): string | null {
  for (const root of repoRoots()) {
    const pointer = resolve(root, 'releases', 'CURRENT')
    if (!existsSync(pointer)) continue
    const name = readFileSync(pointer, 'utf8').trim()
    if (!name) continue
    const dir = resolve(root, 'releases', name)
    if (existsSync(resolve(dir, 'scoring.json'))) return dir
  }
  return null
}

/**
 * Where to load heads from.
 *
 * `models/scoring/` wins so a local export can override what is checked in;
 * otherwise the release that ships with the repository is used, which is why
 * a fresh clone scores without installing anything.
 */
export function defaultScoringDir(): string {
  if (process.env.READALOUDKIT_SCORING) return resolve(process.env.READALOUDKIT_SCORING)
  for (const root of repoRoots()) {
    const dir = resolve(root, 'models', 'scoring')
    if (existsSync(resolve(dir, 'scoring.json'))) return dir
  }
  return shippedReleaseDir() ?? resolve(process.cwd(), 'models', 'scoring')
}

export function scoringReady(dir = defaultScoringDir()): boolean {
  return existsSync(resolve(dir, 'scoring.json'))
}

type Session = import('onnxruntime-node').InferenceSession

/** Rescale a corpus 0-10 score onto the 10-90 range readers expect. */
function toDisplay(raw: number, scale: ScoringManifest['displayScale']): number {
  const [lo, hi] = scale.from
  const [outLo, outHi] = scale.to
  const t = hi === lo ? 0 : (raw - lo) / (hi - lo)
  return Math.max(outLo, Math.min(outHi, outLo + t * (outHi - outLo)))
}

export class CommunityScoringBackend implements ScoringBackend {
  readonly name = 'community'

  get version(): string {
    return this.manifest.version
  }

  private constructor(
    private readonly manifest: ScoringManifest,
    private readonly wordLevel: Session,
    private readonly utterance: Record<string, Session>,
    private readonly utteranceIndex: number[],
  ) {}

  static async load(dir = defaultScoringDir()): Promise<CommunityScoringBackend> {
    const ort = await import('onnxruntime-node')
    const manifest = JSON.parse(
      readFileSync(resolve(dir, 'scoring.json'), 'utf8'),
    ) as ScoringManifest
    const open = (file: string) =>
      ort.InferenceSession.create(resolve(dir, file), { executionProviders: ['cpu'] })

    const wordLevel = await open('word_level.onnx')
    const utterance: Record<string, Session> = {}
    for (const [field, head] of Object.entries(manifest.outputs)) {
      if (field === 'words[].level') continue
      if (!manifest.heads[head]?.shipped) continue
      utterance[field] = await open(manifest.heads[head]!.file)
    }

    // The heads were trained on a subset of the utterance vector; map their key
    // list back onto positions in the full one so the runtime feeds the columns
    // the weights expect.
    const full = [...UTTERANCE_FEATURE_KEYS, ...UTTERANCE_GOP_KEYS] as string[]
    const utteranceIndex = manifest.utteranceFeatureKeys.map((k) => {
      const i = full.indexOf(k)
      if (i < 0) throw new Error(`scoring head wants unknown feature ${k}`)
      return i
    })

    return new CommunityScoringBackend(manifest, wordLevel, utterance, utteranceIndex)
  }

  async score(input: ScoringInput): Promise<ScoringResult | null> {
    if (!input.gopWords.length) return { backend: this.name }
    const ort = await import('onnxruntime-node')
    const levels = this.manifest.wordLevels

    const nWordFeatures = this.manifest.wordFeatureKeys.length
    const flat = new Float32Array(input.gopWords.length * nWordFeatures)
    input.gopWords.forEach((w, i) => flat.set(wordVector(w), i * nWordFeatures))
    const wordOut = await this.wordLevel.run({
      [this.manifest.inputName]: new ort.Tensor('float32', flat, [
        input.gopWords.length,
        nWordFeatures,
      ]),
    })
    const proba = pickProbabilities(wordOut, this.wordLevel.outputNames, levels.length)

    const words: ScoredWord[] = input.gopWords.map((_, i) => {
      const row = proba.slice(i * levels.length, (i + 1) * levels.length)
      const notGood = row.reduce((acc, p, c) => (levels[c] === 'good' ? acc : acc + p), 0)
      const needsAttention = notGood >= ATTENTION_THRESHOLD
      if (!needsAttention) return { level: 'good', needsAttention }
      // Both readings come off the same threshold, so a word can never be shown
      // as `bad` while the flag says it is fine. Within a flagged word the band
      // is the stronger of the two non-good classes.
      let best = 0
      let bestP = -1
      for (let c = 0; c < row.length; c++) {
        if (levels[c] === 'good') continue
        if (row[c]! > bestP) {
          bestP = row[c]!
          best = c
        }
      }
      return { level: levels[best] ?? 'average', needsAttention }
    })

    const agg = utteranceGopFeatures(input.gopWords)
    const fullVector = [...utteranceVector(input.prosody), ...UTTERANCE_GOP_KEYS.map((k) => agg[k])]
    const x = Float32Array.from(this.utteranceIndex.map((i) => fullVector[i] ?? 0))
    const dims = [1, x.length]

    const result: ScoringResult = { backend: this.name, words }
    for (const [field, session] of Object.entries(this.utterance)) {
      const out = await session.run({
        [this.manifest.inputName]: new ort.Tensor('float32', x, dims),
      })
      const raw = Number((out[session.outputNames[0]!]!.data as Float32Array)[0])
      const value = toDisplay(raw, this.manifest.displayScale)
      if (field === 'scores.pronunciation') result.pronunciation = value
      else if (field === 'scores.fluency') result.fluency = value
      else if (field === 'scores.overall') result.overall = value
    }
    return result
  }
}

function pickProbabilities(
  out: Record<string, unknown>,
  names: readonly string[],
  nClasses: number,
): Float32Array {
  for (const name of names) {
    const tensor = out[name] as { data: Float32Array; dims: readonly number[] } | undefined
    if (tensor && tensor.dims.length === 2 && tensor.dims[1] === nClasses) {
      return tensor.data
    }
  }
  throw new Error('word_level graph has no probability output')
}

export function resolveScoringBackend(): ScoringBackend {
  return new NoopScoringBackend()
}

/** The community heads when they are installed, otherwise the noop backend. */
export async function loadScoringBackend(dir = defaultScoringDir()): Promise<ScoringBackend> {
  if (!scoringReady(dir)) return new NoopScoringBackend()
  return CommunityScoringBackend.load(dir)
}
