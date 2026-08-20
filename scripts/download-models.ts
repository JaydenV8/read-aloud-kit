#!/usr/bin/env npx tsx
/**
 * Install the acoustic ONNX (Facebook wav2vec2-base-960h CTC).
 *
 * Resolution order:
 *   1. models/wav2vec2-base-960h-ctc.onnx already installed and valid
 *   2. READALOUDKIT_MODEL_SRC / a local export directory
 *   3. download from the GitHub release asset
 */
import {
  copyFileSync,
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { createHash } from 'node:crypto'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const destDir = resolve(here, '..', 'models')
// The asset lives under its own tag rather than a package release. Replacing
// the file on `v0.1.0` would have made every existing checkout -- which pins the
// previous SHA-256 -- delete its download and fail, so each graph keeps its own
// tag and old clones keep resolving the one they were pinned to.
const expectedName = 'wav2vec2-base-960h-ctc.onnx'
const expectedSha256 = 'a5162c4510b81c13d7c5dab8c80c577caf643bfcada663d74a6fcc9c02b57356'

const modelUrl =
  process.env.READALOUDKIT_MODEL_URL ??
  `https://github.com/JaydenV8/read-aloud-kit/releases/download/acoustic-v2/${expectedName}`

const dest = resolve(destDir, expectedName)

// An explicitly named source is honoured or reported; the scratch directory is
// only a convenience and is skipped when it holds something else. Choosing
// either by existence alone means one stale copy from an earlier export makes
// this command fail the same way forever.
const explicitSrc = process.env.READALOUDKIT_MODEL_SRC
const cacheSrc = '/tmp/rak-models/wav2vec2-base-960h-ctc.onnx' 

function sha256(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

function verify(path: string): void {
  const hash = sha256(path)
  if (hash !== expectedSha256) {
    rmSync(path, { force: true })
    console.error(`SHA256 mismatch for ${expectedName}
  expected ${expectedSha256}
  actual   ${hash}
The partial file was removed. Re-run pnpm models:download.`)
    process.exit(1)
  }
  const labelsDest = resolve(destDir, 'labels.json')
  writeFileSync(
    resolve(destDir, 'INSTALLED.json'),
    JSON.stringify({ file: expectedName, sha256: hash, class: 'EXTERNAL_PRETRAINED' }, null, 2) +
      '\n',
  )
  if (!existsSync(labelsDest)) {
    console.error(`warning: ${labelsDest} is missing; inference will not start without it`)
  }
  console.log(`community acoustic ready  ${dest}  sha256=${hash.slice(0, 12)}…`)
}

async function download(url: string): Promise<void> {
  console.log(`downloading community acoustic model\n  from ${url}`)
  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok || !res.body) {
    console.error(`download failed: HTTP ${res.status} ${res.statusText}
If this repository is still private, the release asset is not publicly readable.
Provide the file another way instead:
  READALOUDKIT_MODEL_SRC=/path/to/${expectedName} pnpm models:download`)
    process.exit(1)
  }
  const total = Number(res.headers.get('content-length') ?? 0)
  let seen = 0
  let lastPct = -1
  const tmp = `${dest}.part`
  const body = Readable.fromWeb(res.body as Parameters<typeof Readable.fromWeb>[0])
  body.on('data', (chunk: Buffer) => {
    seen += chunk.length
    if (!total) return
    const pct = Math.floor((seen / total) * 100)
    if (pct !== lastPct && pct % 5 === 0) {
      lastPct = pct
      process.stdout.write(`  ${pct}%  ${(seen / 1e6).toFixed(0)}/${(total / 1e6).toFixed(0)} MB\r`)
    }
  })
  await pipeline(body, createWriteStream(tmp))
  process.stdout.write('\n')
  copyFileSync(tmp, dest)
  rmSync(tmp, { force: true })
}

mkdirSync(destDir, { recursive: true })

// 1. already installed
if (existsSync(dest) && statSync(dest).size > 0 && sha256(dest) === expectedSha256) {
  verify(dest)
  process.exit(0)
}

// 2. local export
function useLocal(src: string): void {
  console.log(`using local export  ${src}`)
  copyFileSync(src, dest)
  const labelsSrc = resolve(dirname(src), 'labels.json')
  const labelsDest = resolve(destDir, 'labels.json')
  if (existsSync(labelsSrc) && !existsSync(labelsDest)) copyFileSync(labelsSrc, labelsDest)
  verify(dest)
  process.exit(0)
}

if (explicitSrc) {
  if (!existsSync(explicitSrc)) {
    console.error(`READALOUDKIT_MODEL_SRC=${explicitSrc} does not exist`)
    process.exit(1)
  }
  const hash = sha256(explicitSrc)
  if (hash !== expectedSha256) {
    console.error(`READALOUDKIT_MODEL_SRC=${explicitSrc} is not the expected model
  expected ${expectedSha256}
  actual   ${hash}
Re-export it with \`pnpm models:export\`, or unset the variable to download.`)
    process.exit(1)
  }
  useLocal(explicitSrc)
}

if (existsSync(cacheSrc)) {
  const hash = sha256(cacheSrc)
  if (hash === expectedSha256) useLocal(cacheSrc)
  console.log(`ignoring ${cacheSrc}: sha256 ${hash.slice(0, 12)}… is a different model`)
}

// 3. download
await download(modelUrl)
verify(dest)
