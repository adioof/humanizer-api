const http = require('http');
const { humanize, detectOnly } = require('./humanizer');

const PORT = process.env.PORT || 3000;

function parseBody(req, limit = 5 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let body = '';
    let size = 0;
    req.on('data', chunk => {
      size += chunk.length;
      if (size > limit) { req.destroy(); reject(new Error('Body too large')); return; }
      body += chunk;
    });
    req.on('end', () => {
      try { resolve(JSON.parse(body)); }
      catch (e) { reject(new Error('Invalid JSON')); }
    });
    req.on('error', reject);
  });
}

function json(res, data, status = 200) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key',
  });
  res.end(body);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key',
    });
    return res.end();
  }

  // Health
  if (url.pathname === '/' || url.pathname === '/health') {
    return json(res, {
      status: 'ok',
      service: 'humanizer-api',
      version: '1.0.0',
      endpoints: {
        'POST /api/v1/humanize': 'Humanize AI text with LLM rewriting + detection verification',
        'POST /api/v1/detect': 'AI detection only',
      },
      docs: 'https://github.com/adioof/humanizer-api',
    });
  }

  if (req.method !== 'POST') return json(res, { error: 'POST required' }, 405);

  let body;
  try {
    body = await parseBody(req);
  } catch (e) {
    return json(res, { error: e.message }, 400);
  }

  // ============================================================
  // POST /api/v1/humanize
  // ============================================================
  if (url.pathname === '/api/v1/humanize') {
    if (!body.text) return json(res, { error: 'text is required' }, 400);

    // LLM config — accept flat params or nested `llm` object
    const llm = body.llm || {};
    const llmProvider = llm.provider || body.llm_provider || 'openai';
    const llmApiKey = llm.apiKey || body.llm_api_key;
    const llmModel = llm.model || body.llm_model;
    const llmBaseUrl = llm.baseUrl || body.llm_base_url;

    if (!llmApiKey) {
      return json(res, {
        error: 'LLM API key required',
        hint: 'Pass llm_api_key (flat) or llm.apiKey (nested). Supported providers: openai, anthropic, custom',
        example: {
          text: 'Your AI text here...',
          llm_provider: 'openai',
          llm_api_key: 'sk-...',
          llm_model: 'gpt-4o-mini',
          detector_api_key: 'optional-gptzero-key',
        }
      }, 400);
    }

    // Detector config
    const detector = body.detector || {};
    const detectorApiKey = detector.apiKey || body.detector_api_key || process.env.GPTZERO_API_KEY;
    const detectorProvider = detector.provider || body.detector_provider || 'gptzero';

    try {
      const startTime = Date.now();
      const result = await humanize(body.text, {
        llm_provider: llmProvider,
        llm_api_key: llmApiKey,
        llm_model: llmModel,
        llm_base_url: llmBaseUrl,
        detector_provider: detectorProvider,
        detector_api_key: detectorApiKey,
        max_retries: body.max_retries,
        target_score: body.target_score,
        max_chunk_size: body.max_chunk_size,
      });
      result.time_ms = Date.now() - startTime;
      return json(res, result);
    } catch (err) {
      console.error('Humanize error:', err);
      return json(res, { error: err.message }, 500);
    }
  }

  // ============================================================
  // POST /api/v1/detect
  // ============================================================
  if (url.pathname === '/api/v1/detect') {
    if (!body.text) return json(res, { error: 'text is required' }, 400);

    const detector = body.detector || {};
    const apiKey = detector.apiKey || body.detector_api_key || process.env.GPTZERO_API_KEY;
    const provider = detector.provider || body.detector_provider || 'gptzero';

    if (!apiKey) {
      return json(res, {
        error: 'Detector API key required',
        hint: 'Pass detector_api_key or set GPTZERO_API_KEY env var',
      }, 400);
    }

    try {
      const result = await detectOnly(body.text, { provider, apiKey });
      return json(res, result);
    } catch (err) {
      return json(res, { error: err.message }, 500);
    }
  }

  return json(res, { error: 'Not found. Use /api/v1/humanize or /api/v1/detect' }, 404);
});

server.listen(PORT, () => {
  console.log(`Humanizer API running on :${PORT}`);
});
