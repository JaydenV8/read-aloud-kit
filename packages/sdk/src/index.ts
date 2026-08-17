import type { ReadAloudAnalysis } from '@readaloudkit/types'

export class ReadAloudKitClient {
  constructor(private readonly baseUrl: string) {}

  async health(): Promise<unknown> {
    const res = await fetch(`${this.baseUrl.replace(/\/$/, '')}/health`)
    return res.json()
  }

  async analyze(input: { audio: Blob | File; referenceText: string }): Promise<ReadAloudAnalysis> {
    const body = new FormData()
    body.append('referenceText', input.referenceText)
    body.append('audio', input.audio)
    const res = await fetch(`${this.baseUrl.replace(/\/$/, '')}/v1/read-aloud/analyze`, {
      method: 'POST',
      body,
    })
    const data = (await res.json()) as ReadAloudAnalysis & { error?: string }
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
    return data
  }
}
