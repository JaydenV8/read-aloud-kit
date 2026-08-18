import { noCharSeries } from '@readaloudkit/gop'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { GopWord, ProsodyFeatures } from '@readaloudkit/types'
import { acousticFeatures, pausesFromWords, prosodyFromAlignment } from './index.ts'

type Fixture = {
  ref_text: string
  duration: number
  words: { tok: string; t0: number; t1: number }[]
  pauses: { t0: number; t1: number; where: string }[]
  prosody: Record<string, number>
  tone: { sample_rate: number; n: number; hz: number }
  acoustic: Record<string, number>
}

const fixture = JSON.parse(
  readFileSync(resolve(import.meta.dirname, '../../../tests/fixtures/features.json'), 'utf8'),
) as Fixture

/** Only the timings matter here; the acoustic members are not read. */
function asGopWord(w: { tok: string; t0: number; t1: number }): GopWord {
  return {
    tok: w.tok,
    t0: w.t0,
    t1: w.t1,
    gopMean: 0,
    gopMin: 0,
    gopStd: 0,
    marginMean: 0,
    marginMin: 0,
    nFrames: 0,
    nChars: w.tok.length,
    dur: w.t1 - w.t0,
    charPerSec: 0,
    blankRatio: 0,
    logNFrames: 0,
    ...noCharSeries(),
  }
}

const KEY_MAP: Record<string, keyof ProsodyFeatures> = {
  duration: 'duration',
  n_ref: 'nRef',
  n_aligned: 'nAligned',
  coverage: 'coverage',
  spoken_sec: 'spokenSec',
  wpm: 'wpm',
  n_pause: 'nPause',
  pause_total: 'pauseTotal',
  pause_max: 'pauseMax',
  pause_mean: 'pauseMean',
  pause_ratio: 'pauseRatio',
  word_dur_mean: 'wordDurMean',
  word_dur_std: 'wordDurStd',
  word_dur_cv: 'wordDurCv',
  lead_sil: 'leadSil',
  tail_sil: 'tailSil',
  align_ok: 'alignOk',
}

describe('prosody features vs python reference', () => {
  const words = fixture.words.map(asGopWord)

  it('rebuilds the same lead/mid/tail pause list', () => {
    const pauses = pausesFromWords(words, fixture.duration)
    expect(pauses).toEqual(fixture.pauses)
  })

  it('matches every utterance feature', () => {
    const got = prosodyFromAlignment({
      words,
      pauses: pausesFromWords(words, fixture.duration),
      referenceText: fixture.ref_text,
      duration: fixture.duration,
    })
    for (const [pyKey, tsKey] of Object.entries(KEY_MAP)) {
      expect(got[tsKey], tsKey).toBeCloseTo(fixture.prosody[pyKey]!, 10)
    }
  })

  it('reports coverage against words actually spoken, not words aligned', () => {
    const base = {
      words,
      pauses: pausesFromWords(words, fixture.duration),
      referenceText: fixture.ref_text,
      duration: fixture.duration,
    }
    expect(prosodyFromAlignment(base).coverage).toBe(1)
    expect(prosodyFromAlignment({ ...base, spokenWords: 4 }).coverage).toBeCloseTo(0.8, 10)
  })

  it('matches the frame energy statistics', () => {
    const { sample_rate: sr, n, hz } = fixture.tone
    const samples = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      samples[i] = Math.sin((2 * Math.PI * hz * i) / sr) * (i < 1600 ? 0.001 : 0.9)
    }
    const got = acousticFeatures(samples, sr)
    expect(got.rmsMean).toBeCloseTo(fixture.acoustic.rms_mean!, 5)
    expect(got.rmsStd).toBeCloseTo(fixture.acoustic.rms_std!, 5)
    expect(got.rmsPeak).toBeCloseTo(fixture.acoustic.rms_peak!, 5)
    expect(got.activeRatio).toBeCloseTo(fixture.acoustic.active_ratio!, 10)
  })
})
