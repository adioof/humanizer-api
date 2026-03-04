# Humanizer API

LLM-powered AI text humanizer with detection verification loop. Bring your own LLM and detector keys.

## How it works

1. **Chunks** your text (preserves code blocks, tables, markdown)
2. **Rewrites** each prose chunk via your LLM with a battle-tested humanizer prompt
3. **Verifies** with an AI detector (GPTZero, Sapling)
4. If flagged → **targeted rewrite** of flagged sentences
5. Loops until it passes or hits max retries

## API

### `POST /api/v1/humanize`

```json
{
  "text": "Your AI-generated text here...",
  "llm_provider": "openrouter",
  "llm_api_key": "sk-or-...",
  "llm_model": "anthropic/claude-opus-4-6",
  "detector_api_key": "gptzero-key (optional)",
  "max_retries": 2,
  "target_score": 0.3
}
```

**LLM Providers:** `openrouter` (default), `openai`, `anthropic`, `custom` (any OpenAI-compatible endpoint via `llm_base_url`)

**Response:**
```json
{
  "humanized": "The rewritten text...",
  "chunks": 5,
  "detection": {
    "score": 0.12,
    "human_score": 0.88,
    "sentences": []
  },
  "log": [...],
  "time_ms": 8500
}
```

### `POST /api/v1/detect`

```json
{
  "text": "Text to check...",
  "detector_api_key": "gptzero-key"
}
```

## Deploy

```bash
docker build -t humanizer-api .
docker run -p 3000:3000 humanizer-api
```

Or deploy to Railway, Fly.io, etc.

## Self-host

```bash
git clone https://github.com/adioof/humanizer-api
cd humanizer-api
npm install
PORT=3000 node server.js
```
