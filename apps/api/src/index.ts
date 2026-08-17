import { serve } from '@hono/node-server'
import { app } from './app.ts'

const port = Number(process.env.PORT ?? 3000)
console.log(`ReadAloudKit api  http://127.0.0.1:${port}/health`)
serve({ fetch: app.fetch, port, hostname: '127.0.0.1' })
