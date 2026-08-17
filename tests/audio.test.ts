import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { decodeWav, toMono16k } from '@readaloudkit/audio'

describe('wav', () => {
  it('reads the example 16 kHz mono wav', () => {
    const buf = new Uint8Array(readFileSync(resolve(import.meta.dirname, '../examples/sample.wav')))
    const pcm = toMono16k(decodeWav(buf))
    expect(pcm.sampleRate).toBe(16000)
    expect(pcm.samples.length).toBeGreaterThan(1000)
    expect(pcm.durationMs).toBeGreaterThan(500)
  })
})
