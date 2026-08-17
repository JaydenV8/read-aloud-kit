import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { greedyIds, idsToWords } from '@readaloudkit/ctc'
import { alignReference } from './index.ts'

function loadMatrix(name: string): { data: Float32Array; frames: number; vocab: number } {
  const rows = JSON.parse(
    readFileSync(resolve(import.meta.dirname, '../../../tests/fixtures', name), 'utf8'),
  ) as number[][]
  const frames = rows.length
  const vocab = rows[0]!.length
  const data = new Float32Array(frames * vocab)
  for (let t = 0; t < frames; t++) data.set(rows[t]!, t * vocab)
  return { data, frames, vocab }
}

describe('forced alignment against torchaudio goldens', () => {
  it('plants CAT path', () => {
    const gold = JSON.parse(
      readFileSync(resolve(import.meta.dirname, '../../../tests/fixtures/align_cat.json'), 'utf8'),
    ) as { path: number[]; target_ids: number[]; log_probs: number[][] }
    const { frames, vocab } = loadMatrix('align_cat_logprobs.json')
    const stored = new Float32Array(frames * vocab)
    for (let t = 0; t < frames; t++) stored.set(gold.log_probs[t]!, t * vocab)
    const { path } = alignReference({
      logProbs: stored,
      frames,
      vocabSize: vocab,
      referenceTokens: gold.target_ids,
      blankId: 0,
    })
    expect(path).toEqual(gold.path)
  })

  it('sample say wav path matches python', () => {
    const gold = JSON.parse(
      readFileSync(
        resolve(import.meta.dirname, '../../../tests/fixtures/sample_align.json'),
        'utf8',
      ),
    ) as { path: number[]; target_ids: number[] }
    const { data, frames, vocab } = loadMatrix('sample_logprobs.json')
    const { path } = alignReference({
      logProbs: data,
      frames,
      vocabSize: vocab,
      referenceTokens: gold.target_ids,
      blankId: 0,
    })
    expect(path).toEqual(gold.path)
  })

  it('greedy decode of sample matches python', () => {
    const gold = JSON.parse(
      readFileSync(
        resolve(import.meta.dirname, '../../../tests/fixtures/sample_align.json'),
        'utf8',
      ),
    ) as { decode_words: string[] }
    const labels = JSON.parse(
      readFileSync(resolve(import.meta.dirname, '../../../models/labels.json'), 'utf8'),
    ) as { labels: string[] }
    const { data, frames, vocab } = loadMatrix('sample_logprobs.json')
    const ids = greedyIds(data, frames, vocab)
    expect(idsToWords(ids, labels.labels, 0)).toEqual(gold.decode_words)
  })
})
