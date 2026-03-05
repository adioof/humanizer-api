const http = require('http');
const { humanize, detectOnly } = require('./humanizer');
const { detect: detectInHouse } = require('./detector');
const { Auth } = require('./auth');
const { StripeClient, PLANS } = require('./stripe');

const PORT = process.env.PORT || 3000;

// ── Config ──────────────────────────────────────────────────
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;
const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY;
const STRIPE_WEBHOOK_SECRET = process.env.STRIPE_WEBHOOK_SECRET;
const LLM_API_KEY = process.env.LLM_API_KEY;
const GPTZERO_API_KEY = process.env.GPTZERO_API_KEY;
const HF_TOKEN = process.env.HF_TOKEN;
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '*').split(',').map(s => s.trim());
const MAX_TEXT_LENGTH = parseInt(process.env.MAX_TEXT_LENGTH) || 50000;

const auth = (SUPABASE_URL && SUPABASE_SERVICE_KEY) ? new Auth(SUPABASE_URL, SUPABASE_SERVICE_KEY) : null;
const stripe = STRIPE_SECRET_KEY ? new StripeClient(STRIPE_SECRET_KEY) : null;

// ── Route Registry ──────────────────────────────────────────
const routes = [];

function route(method, path, handler) {
  routes.push({ method, path, handler });
}

function matchRoute(method, pathname) {
  return routes.find(r => r.method === method && r.path === pathname);
}

// ── Helpers ─────────────────────────────────────────────────

function parseBody(req, limit = 5 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let body = '';
    let size = 0;
    req.on('data', chunk => {
      size += chunk.length;
      if (size > limit) { req.destroy(); reject(new Error('Body too large')); return; }
      body += chunk;
    });
    req.on('end', () => resolve(body));
    req.on('error', reject);
  });
}

function parseJSON(body) {
  try { return JSON.parse(body); }
  catch { return null; }
}

function json(res, data, status = 200) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': ALLOWED_ORIGINS.includes('*') ? '*' : ALLOWED_ORIGINS[0],
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, DELETE',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key',
  });
  res.end(JSON.stringify(data));
}

function countWords(text) {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

// Extract auth credentials from request headers
function extractAuth(req) {
  const apiKey = req.headers['x-api-key'];
  const authHeader = req.headers['authorization'];
  const bearer = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;
  return { apiKey, bearer };
}

// Resolve user from API key or Bearer token. Returns { userId, keyId } or null.
async function resolveUser(req) {
  if (!auth) return null;
  const { apiKey, bearer } = extractAuth(req);

  if (apiKey) {
    return auth.validateKey(apiKey);
  }
  if (bearer) {
    const userId = await auth.verifyToken(bearer);
    return userId ? { userId, keyId: null } : null;
  }
  return null;
}

// Require auth — sends 401 and returns null if not authenticated
async function requireAuth(req, res) {
  const user = await resolveUser(req);
  if (!user) {
    json(res, { error: 'Authentication required' }, 401);
    return null;
  }
  return user;
}

// Require Bearer token specifically (for key management)
async function requireBearer(req, res) {
  if (!auth) { json(res, { error: 'Auth not configured' }, 503); return null; }
  const { bearer } = extractAuth(req);
  if (!bearer) { json(res, { error: 'Bearer token required' }, 401); return null; }
  const userId = await auth.verifyToken(bearer);
  if (!userId) { json(res, { error: 'Invalid token' }, 401); return null; }
  return userId;
}

// ── Routes: Health ──────────────────────────────────────────

route('GET', '/', (req, res) => {
  json(res, {
    status: 'ok',
    service: 'aucto-humanizer',
    version: '2.0.0',
    auth_enabled: !!auth,
    billing_enabled: !!stripe,
  });
});

route('GET', '/health', (req, res) => json(res, { status: 'ok' }));

// ── Routes: Humanize ────────────────────────────────────────

route('POST', '/api/v1/humanize', async (req, res) => {
  const rawBody = await parseBody(req);
  const body = parseJSON(rawBody);
  if (!body) return json(res, { error: 'Invalid JSON' }, 400);
  if (!body.text || typeof body.text !== 'string') return json(res, { error: 'text is required (string)' }, 400);
  if (body.text.length > MAX_TEXT_LENGTH) return json(res, { error: `Text too long (max ${MAX_TEXT_LENGTH} chars)` }, 400);

  // Auth (optional — BYOK doesn't need it)
  const user = await resolveUser(req);
  const useCredits = !!user;

  // LLM config
  let llmApiKey = body.llm_api_key;
  let llmProvider = body.llm_provider || 'openrouter';
  let llmModel = body.llm_model || 'anthropic/claude-opus-4-6';
  let llmBaseUrl = body.llm_base_url;

  if (!llmApiKey && useCredits && LLM_API_KEY) {
    llmApiKey = LLM_API_KEY;
    llmProvider = 'openrouter';
  }

  if (!llmApiKey) {
    return json(res, {
      error: 'LLM API key required',
      hint: 'Pass llm_api_key (BYOK) or authenticate with x-api-key header to use platform credits',
    }, 400);
  }

  // Credit check
  const wordsIn = countWords(body.text);
  if (useCredits) {
    const credits = await auth.getCredits(user.userId);
    if (credits.balance < wordsIn) {
      return json(res, {
        error: 'Insufficient credits',
        balance: credits.balance,
        required: wordsIn,
      }, 402);
    }
  }

  // Run
  const startTime = Date.now();
  const result = await humanize(body.text, {
    llm_provider: llmProvider,
    llm_api_key: llmApiKey,
    llm_model: llmModel,
    llm_base_url: llmBaseUrl,
    hf_token: body.hf_token || HF_TOKEN,
    detector_provider: body.detector_provider,
    detector_api_key: body.detector_api_key || GPTZERO_API_KEY,
    max_retries: body.max_retries ?? 4,
    target_score: body.target_score ?? 0.98,
    max_chunk_size: body.max_chunk_size,
  });
  result.time_ms = Date.now() - startTime;

  // Deduct + log
  if (useCredits && auth) {
    const wordsOut = countWords(result.humanized);
    const charged = Math.max(wordsIn, wordsOut);
    await auth.deductCredits(user.userId, charged);
    await auth.logUsage({
      userId: user.userId, keyId: user.keyId,
      wordsIn, wordsOut,
      chunks: result.chunks,
      model: llmModel,
      detectionScore: result.detection?.human_score,
      timeMs: result.time_ms,
    });
    result.credits_used = charged;
    const remaining = await auth.getCredits(user.userId);
    result.credits_remaining = remaining.balance;
  }

  json(res, result);
});

// ── Routes: Detect ──────────────────────────────────────────

route('POST', '/api/v1/detect', async (req, res) => {
  const rawBody = await parseBody(req);
  const body = parseJSON(rawBody);
  if (!body?.text) return json(res, { error: 'text is required' }, 400);

  // Default: HuggingFace RoBERTa classifier (free, accurate, no LLM cost)
  const provider = body.detector_provider || body.provider || 'huggingface';

  if (provider === 'huggingface' || provider === 'inhouse') {
    const hfToken = body.hf_token || HF_TOKEN;
    if (!hfToken) return json(res, { error: 'hf_token or HF_TOKEN env var required for detection' }, 400);
    const result = await detectInHouse(body.text, hfToken, { hf_token: hfToken });
    json(res, result);
    return;
  }

  // Third-party detection (gptzero, sapling)
  const detectorKey = body.detector_api_key || GPTZERO_API_KEY;
  if (!detectorKey) return json(res, { error: 'detector_api_key required' }, 400);

  const result = await detectOnly(body.text, {
    provider,
    apiKey: detectorKey,
  });
  json(res, result);
});

// ── Routes: API Keys ────────────────────────────────────────

route('GET', '/api/v1/keys', async (req, res) => {
  const userId = await requireBearer(req, res);
  if (!userId) return;
  const keys = await auth.listKeys(userId);
  json(res, { keys });
});

route('POST', '/api/v1/keys', async (req, res) => {
  const userId = await requireBearer(req, res);
  if (!userId) return;
  const rawBody = await parseBody(req);
  const body = parseJSON(rawBody) || {};
  const name = (typeof body.name === 'string' && body.name.trim()) ? body.name.trim().slice(0, 64) : 'Default';
  const result = await auth.createApiKey(userId, name);
  json(res, { key: result.key, prefix: result.prefix, message: "Save this key — it won't be shown again" }, 201);
});

route('DELETE', '/api/v1/keys', async (req, res) => {
  const userId = await requireBearer(req, res);
  if (!userId) return;
  const rawBody = await parseBody(req);
  const body = parseJSON(rawBody);
  if (!body?.key_id || typeof body.key_id !== 'string') return json(res, { error: 'key_id required (string)' }, 400);
  await auth.revokeKey(userId, body.key_id);
  json(res, { status: 'revoked' });
});

// ── Routes: Credits & Usage ─────────────────────────────────

route('GET', '/api/v1/credits', async (req, res) => {
  const user = await requireAuth(req, res);
  if (!user) return;
  json(res, await auth.getCredits(user.userId));
});

route('GET', '/api/v1/usage', async (req, res) => {
  const user = await requireAuth(req, res);
  if (!user) return;
  json(res, { usage: await auth.getUsage(user.userId) });
});

// ── Routes: Billing ─────────────────────────────────────────

route('GET', '/api/v1/billing/plans', (_req, res) => {
  json(res, { plans: PLANS });
});

route('GET', '/api/v1/pricing', (_req, res) => {
  json(res, {
    free_words: 1000,
    plans: Object.entries(PLANS).map(([key, plan]) => ({
      id: key,
      name: plan.name,
      words: plan.words,
      price_usd: plan.price_cents / 100,
      per_1k_words: ((plan.price_cents / 100) / (plan.words / 1000)).toFixed(2),
    })),
  });
});

route('POST', '/api/v1/billing/checkout', async (req, res) => {
  if (!stripe || !auth) return json(res, { error: 'Billing not configured' }, 503);

  const userId = await requireBearer(req, res);
  if (!userId) return;

  const rawBody = await parseBody(req);
  const body = parseJSON(rawBody);
  if (!body?.plan || !PLANS[body.plan]) {
    return json(res, { error: 'Invalid plan', options: Object.keys(PLANS) }, 400);
  }

  let customerId = await auth.getStripeCustomer(userId);
  if (!customerId) {
    customerId = await stripe.createCustomer(body.email || `user-${userId}@aucto.ai`);
    await auth.saveStripeCustomer(userId, customerId);
  }

  const session = await stripe.createCheckoutSession(
    customerId, body.plan, body.success_url, body.cancel_url,
  );
  json(res, session);
});

route('POST', '/api/v1/billing/webhook', async (req, res) => {
  if (!stripe || !auth) return json(res, { error: 'Billing not configured' }, 503);

  const rawBody = await parseBody(req);
  const sig = req.headers['stripe-signature'];

  if (STRIPE_WEBHOOK_SECRET && sig) {
    if (!stripe.verifyWebhook(rawBody, sig, STRIPE_WEBHOOK_SECRET)) {
      return json(res, { error: 'Invalid signature' }, 400);
    }
  }

  const event = parseJSON(rawBody);
  if (!event) return json(res, { error: 'Invalid JSON' }, 400);

  const checkout = stripe.parseCheckoutEvent(event);
  if (checkout?.words > 0) {
    const userId = await auth.findUserByStripeCustomer(checkout.customerId);
    if (userId) {
      await auth.addCredits(userId, checkout.words);
      await auth.recordPayment(userId, checkout.paymentId, checkout.amountCents, checkout.words);
      console.log(`Credits added: ${checkout.words} words for ${userId}`);
    }
  }

  json(res, { received: true });
});

// ── Server ──────────────────────────────────────────────────

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': ALLOWED_ORIGINS.includes('*') ? '*' : ALLOWED_ORIGINS[0],
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, DELETE',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key',
    });
    return res.end();
  }

  const matched = matchRoute(req.method, url.pathname);
  if (!matched) return json(res, { error: 'Not found' }, 404);

  try {
    await matched.handler(req, res);
  } catch (err) {
    console.error(`Error ${req.method} ${url.pathname}:`, err);
    if (!res.headersSent) {
      json(res, { error: err.message || 'Internal error' }, 500);
    }
  }
});

server.listen(PORT, () => {
  console.log(`aucto.ai Humanizer API v2 on :${PORT}`);
  console.log(`  Auth: ${auth ? '✓' : '✗ (set SUPABASE_URL + SUPABASE_SERVICE_KEY)'}`);
  console.log(`  Billing: ${stripe ? '✓' : '✗ (set STRIPE_SECRET_KEY)'}`);
  console.log(`  Platform LLM: ${LLM_API_KEY ? '✓' : '✗ (BYOK only)'}`);
});
