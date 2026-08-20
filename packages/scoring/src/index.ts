import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { UTTERANCE_FEATURE_KEYS, utteranceVector } from '@readaloudkit/features'
import { UTTERANCE_GOP_KEYS_V2, utteranceGopFeaturesV2, wordVectorBy } from '@readaloudkit/gop'
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
 * At 0.8, `0.5-community` runs precision 0.47 / recall 0.54 and flags about one
 * word in seven, against a true rate of one in eight. Loosening it floods the
 * utterance with low-precision flags; tightening it finds too few of the real
 * problems. See MODEL_CARD.md.
 *
 * It is reported on the test split, which is the one number here selected
 * against data the heads were kept away from. Treat it as a display default to
 * be re-derived if the heads change, not as a measured optimum.
 */
export const ATTENTION_THRESHOLD = 0.8

/**
 * How to turn a pooled hidden layer into features.
 *
 * The projection is fitted on the training corpus, so it belongs to the release
 * and not to the acoustic model -- which stays a stock third-party checkpoint
 * that happens to expose one more tensor. Standardisation and PCA are both
 * affine, so they fold into a single `y = xA + b` and ship as one small graph.
 */
export type HiddenProjection = {
  /** Which transformer layer the weights were fitted on, 1-based. */
  layer: number
  /** Projection for word-pooled vectors. */
  word: string
  /** Projection for the clip-pooled vector; fitted separately. */
  utterance: string
  size: number
  components: number
}

/** Feature keys of the form `hid0`, `hid1`, ... come from the projection. */
const HIDDEN_KEY = /^hid(\d+)$/

export type ScoringManifest = {
  /** Which release these weights are, e.g. `0.5-community`. */
  version: string
  inputName: string
  wordFeatureKeys: string[]
  utteranceFeatureKeys: string[]
  hiddenProjection?: HiddenProjection
  wordLevels: WordLevel[]
  displayScale: { from: [number, number]; to: [number, number] }
  heads: Record<string, { file: string; shipped: boolean }>
  outputs: Record<string, string>
}

/**
 * Split a head's key list into the features read off the GOP/prosody vectors and
 * the count taken from the hidden projection.
 *
 * The hidden keys must be a contiguous tail, numbered from zero. Interleaving
 * them would still work arithmetically but would leave the manifest as the only
 * record of the order, and an order that lives in exactly one place is one
 * nobody can check.
 */
export function splitHiddenKeys(keys: string[], what: string): { plain: string[]; hidden: number } {
  const first = keys.findIndex((k) => HIDDEN_KEY.test(k))
  if (first < 0) return { plain: keys, hidden: 0 }
  const tail = keys.slice(first)
  tail.forEach((k, i) => {
    const m = HIDDEN_KEY.exec(k)
    if (!m || Number(m[1]) !== i) {
      throw new Error(`${what}: hidden feature keys must be a contiguous hid0..hidN tail, got ${k}`)
    }
  })
  return { plain: keys.slice(0, first), hidden: tail.length }
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
    private readonly keys: {
      word: { plain: string[]; hidden: number }
      utterance: { plain: string[]; hidden: number }
    },
    private readonly projection: { word: Session; utterance: Session } | null,
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
    // the weights expect. v2's list is a superset of v1's with the shared keys
    // unchanged, so one lookup table serves either generation.
    const full = [...UTTERANCE_FEATURE_KEYS, ...UTTERANCE_GOP_KEYS_V2] as string[]
    const utteranceIndex = manifest.utteranceFeatureKeys
      .filter((k) => !HIDDEN_KEY.test(k))
      .map((k) => {
        const i = full.indexOf(k)
        if (i < 0) throw new Error(`scoring head wants unknown feature ${k}`)
        return i
      })

    // A head fed a vector of the wrong width does not fail, it scores nonsense.
    // The graph knows what it wants, so check it once here rather than never.
    const widthOf = (s: Session) => {
      const meta = (s as unknown as { inputMetadata?: Record<string, { dimensions?: number[] }> })
        .inputMetadata
      const dims = meta?.[manifest.inputName]?.dimensions
      const last = dims?.[dims.length - 1]
      return typeof last === 'number' && last > 0 ? last : null
    }
    const expect = (s: Session, want: number, what: string) => {
      const got = widthOf(s)
      if (got !== null && got !== want) {
        throw new Error(`${what} wants ${got} features, manifest declares ${want}`)
      }
    }
    expect(wordLevel, manifest.wordFeatureKeys.length, 'word_level.onnx')
    for (const [field, session] of Object.entries(utterance)) {
      expect(session, manifest.utteranceFeatureKeys.length, field)
    }

    const keys = {
      word: splitHiddenKeys(manifest.wordFeatureKeys, 'word_level.onnx'),
      utterance: splitHiddenKeys(manifest.utteranceFeatureKeys, 'utterance heads'),
    }
    const wantsHidden = keys.word.hidden > 0 || keys.utterance.hidden > 0
    const proj = manifest.hiddenProjection
    if (wantsHidden !== Boolean(proj)) {
      throw new Error(
        proj
          ? 'manifest declares a hidden projection but no head asks for hid* features'
          : 'heads ask for hid* features but the manifest declares no hidden projection',
      )
    }
    for (const [what, n] of [
      ['word', keys.word.hidden],
      ['utterance', keys.utterance.hidden],
    ] as const) {
      if (proj && n > 0 && n !== proj.components) {
        throw new Error(`${what} head wants ${n} hidden features, projection emits ${proj.components}`)
      }
    }
    const projection = proj
      ? { word: await open(proj.word), utterance: await open(proj.utterance) }
      : null

    return new CommunityScoringBackend(
      manifest,
      wordLevel,
      utterance,
      utteranceIndex,
      keys,
      projection,
    )
  }

  /**
   * Run pooled vectors through the release's projection.
   *
   * One call for every word rather than one per word: the projection is a
   * single matrix multiply and the per-call overhead dominates it.
   */
  private async project(
    which: 'word' | 'utterance',
    rows: Float32Array[],
    components: number,
  ): Promise<Float32Array> {
    const ort = await import('onnxruntime-node')
    const size = this.manifest.hiddenProjection!.size
    const flat = new Float32Array(rows.length * size)
    rows.forEach((r, i) => flat.set(r, i * size))
    const session = this.projection![which]
    const out = await session.run({
      [session.inputNames[0]!]: new ort.Tensor('float32', flat, [rows.length, size]),
    })
    const data = out[session.outputNames[0]!]!.data as Float32Array
    if (data.length !== rows.length * components) {
      throw new Error(
        `hidden projection returned ${data.length} values, expected ${rows.length * components}`,
      )
    }
    return data
  }

  async score(input: ScoringInput): Promise<ScoringResult | null> {
    if (!input.gopWords.length) return { backend: this.name }
    const ort = await import('onnxruntime-node')
    const levels = this.manifest.wordLevels

    // A release fitted on a hidden layer cannot be served by a graph that does
    // not emit one. Refusing is the only honest answer: filling zeros would
    // return numbers that look like scores and are not.
    if (this.projection && !input.hidden) {
      throw new Error(
        `release ${this.manifest.version} needs hidden layer ${this.manifest.hiddenProjection!.layer};` +
          ' the installed acoustic model does not emit one',
      )
    }
    if (this.projection && input.hidden) {
      const want = this.manifest.hiddenProjection!
      if (input.hidden.size !== want.size) {
        throw new Error(`hidden layer is ${input.hidden.size}-dimensional, projection wants ${want.size}`)
      }
      if (input.hidden.layer !== null && input.hidden.layer !== want.layer) {
        throw new Error(
          `acoustic model emits layer ${input.hidden.layer}, release was fitted on ${want.layer}`,
        )
      }
    }

    const wordHidden =
      this.keys.word.hidden > 0
        ? await this.project('word', input.hidden!.words, this.keys.word.hidden)
        : null

    const nWordFeatures = this.manifest.wordFeatureKeys.length
    const nPlain = this.keys.word.plain.length
    const flat = new Float32Array(input.gopWords.length * nWordFeatures)
    input.gopWords.forEach((w, i) => {
      const base = i * nWordFeatures
      flat.set(wordVectorBy(w, this.keys.word.plain), base)
      if (wordHidden) {
        flat.set(wordHidden.subarray(i * this.keys.word.hidden, (i + 1) * this.keys.word.hidden), base + nPlain)
      }
    })
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

    const agg = utteranceGopFeaturesV2(input.gopWords)
    const fullVector = [
      ...utteranceVector(input.prosody),
      ...UTTERANCE_GOP_KEYS_V2.map((k) => agg[k]),
    ]
    const uttHidden =
      this.keys.utterance.hidden > 0
        ? await this.project('utterance', [input.hidden!.utterance], this.keys.utterance.hidden)
        : null
    const x = new Float32Array(this.manifest.utteranceFeatureKeys.length)
    x.set(this.utteranceIndex.map((i) => fullVector[i] ?? 0))
    if (uttHidden) x.set(uttHidden, this.keys.utterance.plain.length)
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
