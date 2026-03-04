const crypto = require('crypto');
const https = require('https');

// ============================================================
// AUTH + API KEY + CREDITS MANAGEMENT
// Uses Supabase PostgREST API (not raw SQL)
// ============================================================

class Auth {
  constructor(supabaseUrl, supabaseServiceKey) {
    this.url = supabaseUrl.replace(/\/$/, '');
    this.serviceKey = supabaseServiceKey;
  }

  // ── API Keys ──────────────────────────────────────────────

  async createApiKey(userId, name = 'Default') {
    const raw = 'ak_' + crypto.randomBytes(32).toString('hex');
    const hash = crypto.createHash('sha256').update(raw).digest('hex');
    const prefix = raw.slice(0, 11);

    await this._insert('api_keys', {
      user_id: userId,
      key_hash: hash,
      key_prefix: prefix,
      name,
    });

    return { key: raw, prefix };
  }

  async validateKey(key) {
    if (!key || !key.startsWith('ak_')) return null;
    const hash = crypto.createHash('sha256').update(key).digest('hex');

    const rows = await this._select('api_keys', {
      filters: { key_hash: `eq.${hash}`, active: 'eq.true' },
      select: 'id,user_id',
      limit: 1,
    });

    if (!rows.length) return null;

    // Fire-and-forget last_used update
    this._update('api_keys', { last_used_at: new Date().toISOString() }, { id: `eq.${rows[0].id}` }).catch(() => {});

    return { userId: rows[0].user_id, keyId: rows[0].id };
  }

  async listKeys(userId) {
    return this._select('api_keys', {
      filters: { user_id: `eq.${userId}` },
      select: 'id,key_prefix,name,active,created_at,last_used_at',
      order: 'created_at.desc',
    });
  }

  async revokeKey(userId, keyId) {
    await this._update('api_keys', { active: false }, {
      id: `eq.${keyId}`,
      user_id: `eq.${userId}`,
    });
  }

  // ── Credits ───────────────────────────────────────────────

  async getCredits(userId) {
    const rows = await this._select('credits', {
      filters: { user_id: `eq.${userId}` },
      select: 'balance,total_purchased',
      limit: 1,
    });
    return rows[0] || { balance: 0, total_purchased: 0 };
  }

  async deductCredits(userId, words) {
    // Use RPC function for atomic deduction (defined in db.sql)
    const result = await this._rpc('deduct_credits', { p_user_id: userId, p_words: words });
    return result?.success === true;
  }

  async addCredits(userId, words) {
    const result = await this._rpc('add_credits', { p_user_id: userId, p_words: words });
    return result?.success === true;
  }

  // ── Usage Logging ─────────────────────────────────────────

  async logUsage(data) {
    await this._insert('usage_log', {
      user_id: data.userId,
      api_key_id: data.keyId || null,
      words_in: data.wordsIn,
      words_out: data.wordsOut,
      chunks: data.chunks || null,
      llm_model: data.model || null,
      detection_score: data.detectionScore ?? null,
      time_ms: data.timeMs || null,
    });
  }

  async getUsage(userId, limit = 50) {
    return this._select('usage_log', {
      filters: { user_id: `eq.${userId}` },
      select: 'words_in,words_out,chunks,llm_model,detection_score,time_ms,created_at',
      order: 'created_at.desc',
      limit,
    });
  }

  // ── Auth ──────────────────────────────────────────────────

  async verifyToken(token) {
    const result = await this._fetch('/auth/v1/user', {
      headers: { 'Authorization': `Bearer ${token}`, 'apikey': this.serviceKey },
    });
    return result?.id || null;
  }

  // ── Stripe Customers ─────────────────────────────────────

  async getStripeCustomer(userId) {
    const rows = await this._select('stripe_customers', {
      filters: { user_id: `eq.${userId}` },
      select: 'stripe_customer_id',
      limit: 1,
    });
    return rows[0]?.stripe_customer_id || null;
  }

  async saveStripeCustomer(userId, stripeCustomerId) {
    await this._upsert('stripe_customers', {
      user_id: userId,
      stripe_customer_id: stripeCustomerId,
    });
  }

  async findUserByStripeCustomer(stripeCustomerId) {
    const rows = await this._select('stripe_customers', {
      filters: { stripe_customer_id: `eq.${stripeCustomerId}` },
      select: 'user_id',
      limit: 1,
    });
    return rows[0]?.user_id || null;
  }

  async recordPayment(userId, stripePaymentId, amountCents, wordsCredited) {
    await this._insert('payments', {
      user_id: userId,
      stripe_payment_id: stripePaymentId,
      amount_cents: amountCents,
      words_credited: wordsCredited,
      status: 'completed',
    });
  }

  // ── PostgREST Primitives ──────────────────────────────────

  async _select(table, { filters = {}, select = '*', order, limit } = {}) {
    const params = new URLSearchParams();
    params.set('select', select);
    for (const [key, val] of Object.entries(filters)) params.set(key, val);
    if (order) params.set('order', order);
    if (limit) params.set('limit', String(limit));

    const result = await this._fetch(`/rest/v1/${table}?${params.toString()}`);
    return Array.isArray(result) ? result : [];
  }

  async _insert(table, row) {
    return this._fetch(`/rest/v1/${table}`, {
      method: 'POST',
      body: JSON.stringify(row),
      headers: { 'Prefer': 'return=representation' },
    });
  }

  async _update(table, data, filters = {}) {
    const params = new URLSearchParams();
    for (const [key, val] of Object.entries(filters)) params.set(key, val);

    return this._fetch(`/rest/v1/${table}?${params.toString()}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
      headers: { 'Prefer': 'return=representation' },
    });
  }

  async _upsert(table, row) {
    return this._fetch(`/rest/v1/${table}`, {
      method: 'POST',
      body: JSON.stringify(row),
      headers: {
        'Prefer': 'resolution=merge-duplicates,return=representation',
      },
    });
  }

  async _rpc(fn, params = {}) {
    const result = await this._fetch(`/rest/v1/rpc/${fn}`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
    return result;
  }

  // ── HTTP ──────────────────────────────────────────────────

  _fetch(path, opts = {}) {
    const url = new URL(this.url + path);
    const body = opts.body || '';

    return new Promise((resolve, reject) => {
      const headers = {
        'apikey': this.serviceKey,
        'Authorization': `Bearer ${this.serviceKey}`,
        'Content-Type': 'application/json',
        ...(opts.headers || {}),
      };
      if (body) headers['Content-Length'] = String(Buffer.byteLength(body));

      const req = https.request(url, {
        method: opts.method || 'GET',
        headers,
        timeout: 10000,
      }, res => {
        let data = '';
        res.on('data', c => data += c);
        res.on('end', () => {
          if (res.statusCode >= 400) {
            const err = new Error(`Supabase ${res.statusCode}: ${data.slice(0, 300)}`);
            err.status = res.statusCode;
            reject(err);
            return;
          }
          try { resolve(JSON.parse(data)); }
          catch { resolve(data); }
        });
      });
      req.on('error', reject);
      req.on('timeout', () => { req.destroy(); reject(new Error('Supabase timeout')); });
      if (body) req.write(body);
      req.end();
    });
  }
}

module.exports = { Auth };
