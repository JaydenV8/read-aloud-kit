import { readFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { AcousticModel, AcousticOutput } from '@readaloudkit/types'

export type ModelMeta = {
  labels: string[]
  blank: number
  sampleRate: number
  sha256?: string
  onnx?: string
  source?: string
  /** Which transformer layer `hidden` carries, 1-based, when the graph emits one. */
  hidden_layer?: number
}

export function defaultModelDir(): string {
  if (process.env.READALOUDKIT_MODELS) return resolve(process.env.READALOUDKIT_MODELS)
  const here = dirname(fileURLToPath(import.meta.url))
  const candidates = [
    resolve(process.cwd(), 'models'),
    resolve(process.cwd(), '../../models'),
    resolve(here, '../../../../models'),
  ]
  for (const dir of candidates) {
    if (existsSync(resolve(dir, 'labels.json'))) return dir
  }
  return resolve(process.cwd(), 'models')
}

export async function loadMeta(dir = defaultModelDir()): Promise<ModelMeta> {
  const raw = await readFile(resolve(dir, 'labels.json'), 'utf8')
  return JSON.parse(raw) as ModelMeta
}

export function acousticReady(dir = defaultModelDir()): boolean {
  const metaPath = resolve(dir, 'labels.json')
  if (!existsSync(metaPath)) return false
  const onnx = resolve(dir, 'wav2vec2-base-960h-ctc.onnx')
  return existsSync(onnx)
}

export class OnnxAcousticModel implements AcousticModel {
  private constructor(
    private readonly session: import('onnxruntime-node').InferenceSession,
    readonly meta: ModelMeta,
  ) {}

  static async load(dir = defaultModelDir()): Promise<OnnxAcousticModel> {
    const ort = await import('onnxruntime-node')
    const meta = await loadMeta(dir)
    const onnx = resolve(dir, meta.onnx ?? 'wav2vec2-base-960h-ctc.onnx')
    const session = await ort.InferenceSession.create(onnx, {
      executionProviders: ['cpu'],
    })
    return new OnnxAcousticModel(session, meta)
  }

  async infer(audio: Float32Array): Promise<AcousticOutput> {
    const ort = await import('onnxruntime-node')
    const input = new ort.Tensor('float32', audio, [1, audio.length])
    const out = await this.session.run({ waveform: input })
    // By name, not by position. A two-output graph makes the positional read a
    // silent hazard rather than a loud one: a hidden layer has the same rank as
    // the logits and differs only in its last dimension, so reading the wrong
    // one yields a plausible-looking tensor and nonsense downstream.
    const logits = out.logits as import('onnxruntime-node').Tensor | undefined
    if (!logits) {
      throw new Error(`acoustic model has no \`logits\` output (got ${this.session.outputNames.join(', ')})`)
    }
    const dims = logits.dims
    const frames = dims.length === 3 ? dims[1]! : dims[0]!
    const vocabSize = dims.length === 3 ? dims[2]! : dims[1]!
    const result: AcousticOutput = { logits: logits.data as Float32Array, frames, vocabSize }

    const hidden = out.hidden as import('onnxruntime-node').Tensor | undefined
    if (hidden) {
      const hd = hidden.dims
      result.hidden = hidden.data as Float32Array
      result.hiddenSize = hd.length === 3 ? hd[2]! : hd[1]!
      result.hiddenLayer = this.meta.hidden_layer
    }
    return result
  }
}

let cached: Promise<OnnxAcousticModel> | null = null

export function getAcousticModel(dir = defaultModelDir()): Promise<OnnxAcousticModel> {
  cached ??= OnnxAcousticModel.load(dir)
  return cached
}
