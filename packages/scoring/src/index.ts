import type { ScoringBackend, ScoringInput, ScoringResult } from '@readaloudkit/types'

export class NoopScoringBackend implements ScoringBackend {
  readonly name = 'none'
  async score(_input: ScoringInput): Promise<ScoringResult | null> {
    return null
  }
}

export class CommunityScoringBackend implements ScoringBackend {
  readonly name = 'community'
  constructor(private readonly installed: boolean) {}
  async score(_input: ScoringInput): Promise<ScoringResult | null> {
    if (!this.installed) return null
    return null
  }
}

export function resolveScoringBackend(): ScoringBackend {
  return new NoopScoringBackend()
}
