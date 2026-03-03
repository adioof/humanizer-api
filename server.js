const http = require('http');
const { humanize, detectOnly } = require('./humanizer');
const { Auth } = require('./auth');
const { StripeClient, PLANS } = require('./stripe');

const PORT = process.env.PORT || 3000;

// Config
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;
const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY;
const STRIPE_WEBHOOK_SECRET = process.env.STRIPE_WEBHOOK_SECRET;
const LLM_API_KEY = process.env.LLM_API_KEY; // Platform's own OpenAI key for hosted mode
const GPTZERO_API_KEY = process.env.GPTZERO_API_KEY;

const auth = (SUPABASE_URL && SUPABASE_SERVICE_KEY) ? new Auth(SUPABASE_URL, SUPABASE_SERVICE_KEY) : null;
const stripe = STRIPE_SECRET_KEY ? new StripeClient(STRIPE_SECRET_KEY) : null;

// ============================================================
// HELPERS
// ============================================================

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
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, DELETE',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key',
  });
  res.end(body);
}

function countWords(text) {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

// Extract API key or Bearer token from request
function getAuth(req) {
  const apiKey = req.headers['x-api-key'];
  const authHeader = req.headers['authorization'];
  const bearer = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;
  return { apiKey, bearer };
}

// ============================================================
// SERVER
// ============================================================

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  // CORS
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS, DELETE',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key',
    });
    return res.end();
  }

  // Health
  if (url.pathname === '/' || url.pathname === '/health') {
    return json(res, {
      status: 'ok',
      service: 'aucto-humanizer',
      version: '2.0.0',
      auth_enabled: !!auth,
      billing_enabled: !!stripe,
      plans: stripe ? PLANS : undefined,
    });
  }

  // ============================================================
  // PUBLIC: Humanize (BYOK or authenticated)
  // ============================================================
  if (url.pathname === '/api/v1/humanize' && req.method === 'POST') {
    const rawBody = await parseBody(req);
    const body = parseJSON(rawBody);
    if (!body) return json(res, { error: 'Invalid JSON' }, 400);
    if (!body.text) return json(res, { error: 'text is required' }, 400);

    const { apiKey, bearer } = getAuth(req);
    let userId = null, keyId = null, useCredits = false;

    // Mode 1: Authenticated with API key (uses platform credits)
    if (apiKey && auth) {
      const keyData = await auth.validateKey(apiKey);
      if (!keyData) return json(res, { error: 'Invalid API key' }, 401);
      userId = keyData.userId;
      keyId = keyData.keyId;
      useCredits = true;
    }
    // Mode 2: Authenticated with Bearer token (uses platform credits)
    else if (bearer && auth) {
      userId = await auth.verifyToken(bearer);
      if (!userId) return json(res, { error: 'Invalid token' }, 401);
      useCredits = true;
    }

    // Determine LLM config
    let llmApiKey = body.llm_api_key || body.llm?.apiKey;
    let llmProvider = body.llm_provider || body.llm?.provider || 'openai';
    let llmModel = body.llm_model || body.llm?.model || 'gpt-4o-mini';
    let llmBaseUrl = body.llm_base_url || body.llm?.baseUrl;

    // If authenticated and no BYOK key provided, use platform key
    if (!llmApiKey && useCredits && LLM_API_KEY) {
      llmApiKey = LLM_API_KEY;
      llmProvider = 'openai';
    }

    if (!llmApiKey) {
      return json(res, {
        error: 'LLM API key required',
        hint: 'Either pass llm_api_key (BYOK) or authenticate with an API key (x-api-key header) to use platform credits',
      }, 400);
    }

    // Check credits if using platform
    const wordsIn = countWords(body.text);
    if (useCredits) {
      const credits = await auth.getCredits(userId);
      if (credits.balance < wordsIn) {
        return json(res, {
          error: 'Insufficient credits',
          balance: credits.balance,
          required: wordsIn,
          hint: 'Purchase more credits at POST /api/v1/billing/checkout',
        }, 402);
      }
    }

    // Run humanizer
    try {
      const startTime = Date.now();
      const result = await humanize(body.text, {
        llm_provider: llmProvider,
        llm_api_key: llmApiKey,
        llm_model: llmModel,
        llm_base_url: llmBaseUrl,
        detector_provider: body.detector_provider,
        detector_api_key: body.detector_api_key || GPTZERO_API_KEY,
        max_retries: body.max_retries,
        target_score: body.target_score,
        max_chunk_size: body.max_chunk_size,
      });
      result.time_ms = Date.now() - startTime;

      // Deduct credits + log usage
      const wordsOut = countWords(result.humanized);
      if (useCredits && auth) {
        const charged = Math.max(wordsIn, wordsOut);
        await auth.deductCredits(userId, charged);
        await auth.logUsage({
          userId, keyId, wordsIn, wordsOut,
          chunks: result.chunks,
          model: llmModel,
          detectionScore: result.detection?.score,
          timeMs: result.time_ms,
        });
        result.credits_used = charged;
        const remaining = await auth.getCredits(userId);
        result.credits_remaining = remaining.balance;
      }

      return json(res, result);
    } catch (err) {
      console.error('Humanize error:', err);
      return json(res, { error: err.message }, 500);
    }
  }

  // ============================================================
  // PUBLIC: Detect only
  // ============================================================
  if (url.pathname === '/api/v1/detect' && req.method === 'POST') {
    const rawBody = await parseBody(req);
    const body = parseJSON(rawBody);
    if (!body?.text) return json(res, { error: 'text is required' }, 400);

    const detectorKey = body.detector_api_key || GPTZERO_API_KEY;
    if (!detectorKey) return json(res, { error: 'detector_api_key required' }, 400);

    try {
      const result = await detectOnly(body.text, {
        provider: body.detector_provider || 'gptzero',
        apiKey: detectorKey,
      });
      return json(res, result);
    } catch (err) {
      return json(res, { error: err.message }, 500);
    }
  }

  // ============================================================
  // AUTH: API Keys
  // ============================================================
  if (url.pathname === '/api/v1/keys' && auth) {
    const { bearer } = getAuth(req);
    const userId = bearer ? await auth.verifyToken(bearer) : null;
    if (!userId) return json(res, { error: 'Authentication required (Bearer token)' }, 401);

    if (req.method === 'GET') {
      const keys = await auth.listKeys(userId);
      return json(res, { keys });
    }
    if (req.method === 'POST') {
      const rawBody = await parseBody(req);
      const body = parseJSON(rawBody) || {};
      const result = await auth.createApiKey(userId, body.name);
      return json(res, {
        key: result.key,
        prefix: result.prefix,
        message: 'Save this key — it won\'t be shown again',
      }, 201);
    }
    if (req.method === 'DELETE') {
      const rawBody = await parseBody(req);
      const body = parseJSON(rawBody);
      if (!body?.key_id) return json(res, { error: 'key_id required' }, 400);
      await auth.revokeKey(userId, body.key_id);
      return json(res, { status: 'revoked' });
    }
  }

  // ============================================================
  // AUTH: Credits + Usage
  // ============================================================
  if (url.pathname === '/api/v1/credits' && req.method === 'GET' && auth) {
    const { bearer, apiKey } = getAuth(req);
    let userId;
    if (apiKey) {
      const keyData = await auth.validateKey(apiKey);
      userId = keyData?.userId;
    } else if (bearer) {
      userId = await auth.verifyToken(bearer);
    }
    if (!userId) return json(res, { error: 'Authentication required' }, 401);

    const credits = await auth.getCredits(userId);
    return json(res, credits);
  }

  if (url.pathname === '/api/v1/usage' && req.method === 'GET' && auth) {
    const { bearer, apiKey } = getAuth(req);
    let userId;
    if (apiKey) {
      const keyData = await auth.validateKey(apiKey);
      userId = keyData?.userId;
    } else if (bearer) {
      userId = await auth.verifyToken(bearer);
    }
    if (!userId) return json(res, { error: 'Authentication required' }, 401);

    const usage = await auth.getUsage(userId);
    return json(res, { usage });
  }

  // ============================================================
  // BILLING: Stripe
  // ============================================================
  if (url.pathname === '/api/v1/billing/plans' && req.method === 'GET') {
    return json(res, { plans: PLANS });
  }

  if (url.pathname === '/api/v1/billing/checkout' && req.method === 'POST' && stripe && auth) {
    const { bearer } = getAuth(req);
    const userId = bearer ? await auth.verifyToken(bearer) : null;
    if (!userId) return json(res, { error: 'Authentication required' }, 401);

    const rawBody = await parseBody(req);
    const body = parseJSON(rawBody);
    if (!body?.plan) return json(res, { error: 'plan required', options: Object.keys(PLANS) }, 400);

    try {
      // Get or create Stripe customer
      let customerId = await auth.getOrCreateStripeCustomer(userId);
      if (!customerId) {
        // Need email from Supabase auth
        customerId = await stripe.createCustomer(body.email || `user-${userId}@aucto.ai`);
        await auth.saveStripeCustomer(userId, customerId);
      }

      const session = await stripe.createCheckoutSession(
        customerId,
        body.plan,
        body.success_url,
        body.cancel_url,
      );

      return json(res, session);
    } catch (err) {
      return json(res, { error: err.message }, 500);
    }
  }

  // Stripe webhook
  if (url.pathname === '/api/v1/billing/webhook' && req.method === 'POST' && stripe && auth) {
    const rawBody = await parseBody(req);
    const sig = req.headers['stripe-signature'];

    if (STRIPE_WEBHOOK_SECRET && sig) {
      const valid = stripe.verifyWebhook(rawBody, sig, STRIPE_WEBHOOK_SECRET);
      if (!valid) return json(res, { error: 'Invalid signature' }, 400);
    }

    const event = parseJSON(rawBody);
    if (!event) return json(res, { error: 'Invalid JSON' }, 400);

    const checkout = stripe.parseCheckoutEvent(event);
    if (checkout && checkout.words > 0) {
      // Find user by Stripe customer ID
      // Add credits
      try {
        // Look up user by stripe customer id
        const userId = await auth._query(
          `select user_id from stripe_customers where stripe_customer_id = $1`,
          [checkout.customerId]
        ).then(r => r[0]?.user_id);

        if (userId) {
          await auth.addCredits(userId, checkout.words);
          await auth.recordPayment(userId, checkout.paymentId, checkout.amountCents, checkout.words);
          console.log(`Credits added: ${checkout.words} words for user ${userId}`);
        }
      } catch (err) {
        console.error('Webhook processing error:', err);
      }
    }

    return json(res, { received: true });
  }

  // ============================================================
  // PRICING PAGE DATA
  // ============================================================
  if (url.pathname === '/api/v1/pricing' && req.method === 'GET') {
    return json(res, {
      free_words: 1000,
      plans: Object.entries(PLANS).map(([key, plan]) => ({
        id: key,
        name: plan.name,
        words: plan.words,
        price_usd: plan.price_cents / 100,
        per_1k_words: ((plan.price_cents / 100) / (plan.words / 1000)).toFixed(2),
      })),
    });
  }

  return json(res, { error: 'Not found' }, 404);
});

server.listen(PORT, () => {
  console.log(`aucto.ai Humanizer API v2 running on :${PORT}`);
  console.log(`Auth: ${auth ? 'enabled' : 'disabled (set SUPABASE_URL + SUPABASE_SERVICE_KEY)'}`);
  console.log(`Billing: ${stripe ? 'enabled' : 'disabled (set STRIPE_SECRET_KEY)'}`);
  console.log(`Platform LLM: ${LLM_API_KEY ? 'enabled' : 'disabled (BYOK only)'}`);
});
