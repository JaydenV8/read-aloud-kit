import { Hono } from 'hono'
import { cors } from 'hono/cors'
import type { ContentfulStatusCode } from 'hono/utils/http-status'
import { zValidator } from '@hono/zod-validator'
import { z } from 'zod'
import { ReadAloudAnalyzer } from '@readaloudkit/core'
import { PACKAGE_VERSION } from '@readaloudkit/types'

const analyzer = new ReadAloudAnalyzer()

const MAX_AUDIO_BYTES = 20 * 1024 * 1024
const MAX_REFERENCE_CHARS = 5000

/**
 * `text` is accepted as a legacy alias for `referenceText`; exactly one of them
 * has to carry a non-empty prompt.
 */
const analyzeForm = z
  .object({
    audio: z
      .instanceof(File, { message: 'audio file is required' })
      .refine((f) => f.size > 0, { message: 'audio file is empty' })
      .refine((f) => f.size <= MAX_AUDIO_BYTES, { message: 'audio larger than 20MB' }),
    referenceText: z.string().max(MAX_REFERENCE_CHARS).optional(),
    text: z.string().max(MAX_REFERENCE_CHARS).optional(),
  })
  .transform((value, ctx) => {
    const referenceText = (value.referenceText ?? value.text ?? '').trim()
    if (!referenceText) {
      ctx.addIssue({
        code: 'custom',
        message: 'referenceText is required',
        path: ['referenceText'],
      })
      return z.NEVER
    }
    return { audio: value.audio, referenceText }
  })

export const app = new Hono()

app.use('*', cors())

app.get('/health', async (c) => {
  const ready = analyzer.ready()
  return c.json({
    ok: true,
    ready,
    package: PACKAGE_VERSION,
    acoustic: ready ? 'community' : 'missing',
    scoring: await analyzer.scoringBackend(),
  })
})

app.post(
  '/v1/read-aloud/analyze',
  // Checked before the body is read so an unconfigured server fails fast.
  async (c, next) => {
    if (!analyzer.ready()) {
      return c.json(
        {
          ok: false,
          error: 'Community acoustic model is not installed. Run pnpm models:download.',
        },
        503,
      )
    }
    await next()
  },
  zValidator('form', analyzeForm, (result, c) => {
    if (!result.success) {
      const issues = result.error.issues.map((i) => ({
        field: i.path.join('.') || 'body',
        message: i.message,
      }))
      return c.json({ ok: false, error: issues[0]?.message ?? 'invalid request', issues }, 400)
    }
  }),
  async (c) => {
    try {
      const { audio, referenceText } = c.req.valid('form')
      const result = await analyzer.analyze({
        audio: new Uint8Array(await audio.arrayBuffer()),
        referenceText,
        mime: audio.type || 'audio/wav',
      })
      return c.json(result)
    } catch (e) {
      // Only a message something deliberately tagged with a status is handed
      // back. An untagged throw came out of a library — onnxruntime's load
      // failures, for one, quote the absolute path of the model file — and
      // that belongs in the server's log, not in a reply to whoever asked.
      const err = e as Error & { status?: number }
      if (typeof err.status === 'number') {
        return c.json({ ok: false, error: err.message }, err.status as ContentfulStatusCode)
      }
      console.error('analyze failed:', err)
      return c.json({ ok: false, error: 'analyze failed', code: 'internal' }, 500)
    }
  },
)
