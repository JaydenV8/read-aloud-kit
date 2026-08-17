import type { AnalysisWord, PauseSpan, ProsodyFeatures } from '@readaloudkit/types'

/** Words per minute above which the pace is called out. */
export const FAST_WPM = 180
/** Share of the prompt that may go unread before coverage itself is the advice. */
export const COVERAGE_SLACK = 0.15
/** Only the first few offenders are named; a wall of words is not advice. */
const MAX_NAMED = 5

export type TipsInput = {
  words: AnalysisWord[]
  prosody: ProsodyFeatures
  pauses: PauseSpan[]
  /** Words a pronunciation head marked poor. Empty without a scoring backend. */
  lowLevelWords?: string[]
  /** Function words a weak-form head says were over-articulated. */
  weakFormWords?: string[]
}

function names(words: AnalysisWord[], want: AnalysisWord['status']): string[] {
  return words
    .filter((w) => w.status === want)
    .map((w) => w.reference ?? w.hypothesis ?? '')
    .filter(Boolean)
}

function list(items: string[]): string {
  return items.slice(0, MAX_NAMED).join(', ')
}

/**
 * Rule-driven advice built from the error list. Deliberately not generative:
 * every line points at something the analysis actually measured.
 */
export function buildTips(input: TipsInput): string[] {
  const { words, prosody, pauses } = input
  const tips: string[] = []

  const omitted = names(words, 'omission')
  const inserted = names(words, 'insertion')
  const repeated = names(words, 'repetition')
  const replaced = words
    .filter((w) => w.status === 'substitution')
    .map((w) => (w.hypothesis ? `${w.reference} → ${w.hypothesis}` : (w.reference ?? '')))
    .filter(Boolean)

  if (omitted.length) {
    tips.push(`Barely spoken: ${list(omitted)}. Every omitted word costs a content point.`)
  }
  if (replaced.length) {
    tips.push(`Came out as a different word: ${list(replaced)}. Check these against the prompt.`)
  }
  if (inserted.length) {
    tips.push(`Extra words: ${list(inserted)}. Read the prompt as written.`)
  }
  if (repeated.length) {
    tips.push(
      `Repeated: ${list(repeated)}. Repetition is not a content error, but it costs fluency.`,
    )
  }
  if (input.lowLevelWords?.length) {
    tips.push(`Pronounced weakly: ${list(input.lowLevelWords)}. Read them again slowly.`)
  }
  if (input.weakFormWords?.length) {
    tips.push(`Function words are too heavy: ${list(input.weakFormWords)}. Let them pass lightly.`)
  }

  const longest = pauses.reduce((max, p) => (p.durMs > max ? p.durMs : max), 0)
  if (longest > 0) {
    tips.push(
      `Longest pause is about ${(longest / 1000).toFixed(2)}s. Keep phrases connected and avoid breathing mid-phrase.`,
    )
  }
  if (prosody.wpm > FAST_WPM) {
    tips.push('Your pace is fast. Break at sense groups instead of pushing through every word.')
  }
  if (prosody.nRef > 0 && omitted.length / prosody.nRef > COVERAGE_SLACK) {
    tips.push('Coverage is incomplete. Get every word spoken before working on fluency.')
  }

  if (!tips.length) {
    tips.push('Overall steady. Keep reading by sense groups and keep the content words clear.')
  }
  return tips
}
