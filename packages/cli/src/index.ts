import { readFile } from 'node:fs/promises'
import { ReadAloudAnalyzer } from '@readaloudkit/core'

async function main() {
  const args = process.argv.slice(2).filter((a) => a !== '--')
  const cmd = args[0] === 'analyze' ? 'analyze' : args.includes('--audio') ? 'analyze' : args[0]
  if (cmd !== 'analyze') {
    console.error('usage: readaloudkit analyze --audio <wav> --text <ref>')
    process.exit(1)
  }
  if (args[0] === 'analyze') args.shift()
  let audio = ''
  let text = ''
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--audio') audio = args[++i] ?? ''
    else if (args[i] === '--text') text = args[++i] ?? ''
  }
  if (!audio || !text) {
    console.error('usage: readaloudkit analyze --audio <wav> --text <ref>')
    process.exit(1)
  }
  const buf = new Uint8Array(await readFile(audio))
  const result = await new ReadAloudAnalyzer().analyze({ audio: buf, referenceText: text })
  console.log(JSON.stringify(result, null, 2))
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
