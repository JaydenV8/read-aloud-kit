# Security

- Do not commit `.env`, cookies, API tokens, or internal hostnames.
- Do not commit model weights. They are fetched by `pnpm models:download` and verified against a pinned SHA-256.
- The acoustic ONNX is a public Facebook ASR export, not a secret.
- The HTTP API keeps uploads in memory, or in a temp file when ffmpeg has to convert them, and deletes that file afterwards. Nothing goes to a durable store.
- Speech is personal data in most jurisdictions. If you deploy this, decide where recordings go before you collect any.
- If you find credentials in a clone, rotate them. Do not print the secret.
