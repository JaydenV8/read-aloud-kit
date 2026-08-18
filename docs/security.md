# Security

- Do not commit `.env`, cookies, API tokens, or internal hostnames.
- Do not commit the acoustic model. It is 378 MB, fetched by `pnpm models:download` and verified against a pinned SHA-256. It is a public Facebook ASR export, not a secret.
- The scoring heads under `releases/` are committed on purpose: they are about a megabyte, they are trained on a public corpus, and they are published under CC BY 4.0. Nothing in them is derived from private data.
- The HTTP API keeps uploads in memory, or in a temp file when ffmpeg has to convert them, and deletes that file afterwards. Nothing goes to a durable store.
- Speech is personal data in most jurisdictions. If you deploy this, decide where recordings go before you collect any.
- If you find credentials in a clone, rotate them. Do not print the secret.
- The HTTP API returns a fixed body for internal failures rather than the exception text. Library errors quote local filesystem paths; those go to the server log.
