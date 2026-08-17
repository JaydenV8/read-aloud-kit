import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  contentFromEdits,
  countEdits,
  dualTrackTokens,
  editsFromTexts,
  nearPhone,
  softenOps,
  wordEdits,
} from './index.ts'

const fixtures = JSON.parse(
  readFileSync(resolve(import.meta.dirname, '../../../tests/fixtures/edits.json'), 'utf8'),
) as Record<
  string,
  { n_omission: number; n_replacement: number; n_insertion: number; content: number }
>

describe('word edits and content scoring', () => {
  it('perfect match', () => {
    const ops = wordEdits('the cat sat', 'the cat sat')
    const c = countEdits(ops)
    expect(c.nEqual).toBe(3)
    expect(c.nContentErrors).toBe(0)
    expect(editsFromTexts('the cat sat', 'the cat sat').content).toBe(5)
  })

  it('omit replace insert raw', () => {
    const ops = wordEdits('the cat sat on the mat', 'the dog sat the mat here')
    const kinds = ops.map((o) => o.op)
    expect(kinds).toContain('replace')
    expect(kinds).toContain('omit')
    expect(kinds).toContain('insert')
    const c = countEdits(ops)
    expect(c.nOmission).toBe(1)
    expect(c.nReplacement).toBe(1)
    expect(c.nInsertion).toBe(1)
    expect(
      editsFromTexts('the cat sat on the mat', 'the dog sat the mat here', false).content,
    ).toBe(contentFromEdits(6, 1, 1, 1))
  })

  it('repeat is not content', () => {
    const c = countEdits(wordEdits('go home', 'go go home'))
    expect(c.nRepeat).toBe(1)
    expect(c.nInsertion).toBe(0)
    expect(editsFromTexts('go home', 'go go home').content).toBe(5)
  })

  it('true insert still counts raw', () => {
    const c = countEdits(wordEdits('go home', 'go now home'))
    expect(c.nInsertion).toBe(1)
  })

  it('all omitted', () => {
    const ed = editsFromTexts('one two three four', '')
    expect(ed.nOmission).toBe(4)
    expect(ed.content).toBe(0)
  })

  it('near phone and function replace', () => {
    expect(nearPhone('effect', 'affect')).toBe(true)
    expect(nearPhone('the', 'a')).toBe(true)
    expect(nearPhone('cat', 'dog')).toBe(false)
    expect(editsFromTexts('the effect was clear', 'a affect was clear').nReplacement).toBe(0)
    expect(editsFromTexts('the cat sat', 'the dog sat').nReplacement).toBe(1)
  })

  it('long tail omit', () => {
    const ed = editsFromTexts('one two three four five six seven', 'one two three')
    expect(ed.nOmission).toBe(4)
  })

  it('short mid omit not content', () => {
    const ed = editsFromTexts('the cat sat on the mat', 'the cat sat the mat')
    expect(ed.nOmission).toBe(0)
    expect(ed.content).toBe(5)
  })

  it('insert run counts', () => {
    expect(editsFromTexts('go home', 'go really fast home').nInsertion).toBe(2)
    expect(countEdits(softenOps(wordEdits('go home', 'go now home'))).nInsertion).toBe(0)
  })

  it('dual track long tail', () => {
    const ref = 'one two three four five six seven'
    const align = ref.split(' ').map((tok) => ({ tok }))
    const { tokens, counts } = dualTrackTokens(align, ref, 'one two three')
    expect(tokens.filter((t) => t.kind === 'omission').map((t) => t.display)).toEqual([
      'four',
      'five',
      'six',
      'seven',
    ])
    expect(counts.nOmission).toBe(4)
  })

  it('matches python fixtures', () => {
    const pairs: [string, [string, string, boolean?]][] = [
      ['perfect', ['the cat sat', 'the cat sat']],
      ['omit_replace_insert_raw', ['the cat sat on the mat', 'the dog sat the mat here', false]],
      ['repeat', ['go home', 'go go home']],
      ['long_tail', ['one two three four five six seven', 'one two three']],
      ['short_mid_omit', ['the cat sat on the mat', 'the cat sat the mat']],
      ['near_phone', ['the effect was clear', 'a affect was clear']],
      ['cat_dog', ['the cat sat', 'the dog sat']],
      ['insert_run', ['go home', 'go really fast home']],
    ]
    for (const [key, args] of pairs) {
      const ed = editsFromTexts(args[0], args[1], args[2] ?? true)
      const gold = fixtures[key]!
      expect(ed.nOmission, key).toBe(gold.n_omission)
      expect(ed.nReplacement, key).toBe(gold.n_replacement)
      expect(ed.nInsertion, key).toBe(gold.n_insertion)
      expect(ed.content, key).toBeCloseTo(gold.content, 5)
    }
  })
})
