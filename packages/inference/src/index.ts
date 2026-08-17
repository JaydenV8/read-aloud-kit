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
    const first = out[this.session.outputNames[0]!] as import('onnxruntime-node').Tensor
    const data = first.data as Float32Array
    const dims = first.dims
    const frames = dims.length === 3 ? dims[1]! : dims[0]!
    const vocabSize = dims.length === 3 ? dims[2]! : dims[1]!
    return { logits: data, frames, vocabSize }
  }
}

let cached: Promise<OnnxAcousticModel> | null = null

export function getAcousticModel(dir = defaultModelDir()): Promise<OnnxAcousticModel> {
  cached ??= OnnxAcousticModel.load(dir)
  return cached
}
