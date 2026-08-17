export type Pcm = {
  samples: Float32Array
  sampleRate: number
  durationMs: number
}

function readU32(v: DataView, o: number) {
  return v.getUint32(o, true)
}
function readU16(v: DataView, o: number) {
  return v.getUint16(o, true)
}

export function decodeWav(buf: Uint8Array): Pcm {
  if (buf.byteLength < 12) throw new Error('audio too small')
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const tag = String.fromCharCode(buf[0]!, buf[1]!, buf[2]!, buf[3]!)
  if (tag !== 'RIFF')
    throw new Error('only WAV is parsed in-process; install ffmpeg for other types')
  let offset = 12
  let sampleRate = 16000
  let channels = 1
  let bits = 16
  let format = 1
  let dataOff = -1
  let dataLen = 0
  while (offset + 8 <= buf.byteLength) {
    const id = String.fromCharCode(
      buf[offset]!,
      buf[offset + 1]!,
      buf[offset + 2]!,
      buf[offset + 3]!,
    )
    const size = readU32(view, offset + 4)
    const start = offset + 8
    if (id === 'fmt ') {
      format = readU16(view, start)
      channels = readU16(view, start + 2)
      sampleRate = readU32(view, start + 4)
      bits = readU16(view, start + 14)
    } else if (id === 'data') {
      dataOff = start
      dataLen = size
      break
    }
    offset = start + size + (size % 2)
  }
  if (dataOff < 0) throw new Error('WAV missing data chunk')
  const n = Math.floor(dataLen / ((bits / 8) * channels))
  const mono = new Float32Array(n)
  if (format === 1 && bits === 16) {
    for (let i = 0; i < n; i++) {
      let acc = 0
      for (let c = 0; c < channels; c++) {
        acc += view.getInt16(dataOff + (i * channels + c) * 2, true) / 32768
      }
      mono[i] = acc / channels
    }
  } else if ((format === 3 || format === 1) && bits === 32) {
    for (let i = 0; i < n; i++) {
      let acc = 0
      for (let c = 0; c < channels; c++) {
        acc +=
          format === 3
            ? view.getFloat32(dataOff + (i * channels + c) * 4, true)
            : view.getInt32(dataOff + (i * channels + c) * 4, true) / 2147483648
      }
      mono[i] = acc / channels
    }
  } else {
    throw new Error(`unsupported WAV format=${format} bits=${bits}`)
  }
  return { samples: mono, sampleRate, durationMs: (n / sampleRate) * 1000 }
}

export function resample(samples: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return samples
  const n = Math.max(1, Math.round((samples.length * to) / from))
  const out = new Float32Array(n)
  const ratio = from / to
  for (let i = 0; i < n; i++) {
    const x = i * ratio
    const i0 = Math.floor(x)
    const i1 = Math.min(samples.length - 1, i0 + 1)
    const t = x - i0
    out[i] = samples[i0]! * (1 - t) + samples[i1]! * t
  }
  return out
}

export function toMono16k(pcm: Pcm): Pcm {
  const samples = resample(pcm.samples, pcm.sampleRate, 16000)
  return { samples, sampleRate: 16000, durationMs: (samples.length / 16000) * 1000 }
}

export async function maybeFfmpegToWav(input: Uint8Array, mime: string): Promise<Uint8Array> {
  if (mime.includes('wav') || mime.includes('wave')) return input
  const { spawn } = await import('node:child_process')
  const { tmpdir } = await import('node:os')
  const { join } = await import('node:path')
  const { writeFile, readFile, rm } = await import('node:fs/promises')
  const dir = join(tmpdir(), `rak-${Date.now()}`)
  const { mkdir } = await import('node:fs/promises')
  await mkdir(dir, { recursive: true })
  const src = join(dir, 'in.bin')
  const dest = join(dir, 'out.wav')
  await writeFile(src, input)
  await new Promise<void>((resolve, reject) => {
    const p = spawn(
      'ffmpeg',
      ['-y', '-i', src, '-ac', '1', '-ar', '16000', '-sample_fmt', 's16', dest],
      { stdio: 'ignore' },
    )
    p.on('error', () => reject(new Error('ffmpeg is required for non-WAV audio')))
    p.on('exit', (code) => (code === 0 ? resolve() : reject(new Error('ffmpeg failed'))))
  })
  const out = await readFile(dest)
  await rm(dir, { recursive: true, force: true })
  return new Uint8Array(out)
}
