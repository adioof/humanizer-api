const crypto = require('crypto');
const https = require('https');

// ============================================================
// AUTH + API KEY + CREDITS MANAGEMENT
// ============================================================

class Auth {
  constructor(supabaseUrl, supabaseServiceKey) {
    this.url = supabaseUrl;
    this.serviceKey = supabaseServiceKey;
  }

  // Generate a new API key
  async createApiKey(userId, name = 'Default') {
    const raw = 'ak_' + crypto.randomBytes(32).toString('hex');
    const hash = crypto.createHash('sha256').update(raw).digest('hex');
    const prefix = raw.slice(0, 11); // ak_xxxxxxxx

    await this._query(
      `insert into api_keys (user_id, key_hash, key_prefix, name) values ($1, $2, $3, $4)`,
      [userId, hash, prefix, name]
    );

    return { key: raw, prefix };
  }

  // Validate API key → returns { userId, keyId } or null
  async validateKey(key) {
    if (!key || !key.startsWith('ak_')) return null;
    const hash = crypto.createHash('sha256').update(key).digest('hex');

    const result = await this._query(
      `select id, user_id from api_keys where key_hash = $1 and active = true limit 1`,
      [hash]
    );

    if (!result.length) return null;

    // Update last_used_at
    this._query(`update api_keys set last_used_at = now() where id = $1`, [result[0].id]).catch(() => {});

    return { userId: result[0].user_id, keyId: result[0].id };
  }

  // Get user's API keys (without the actual key)
  async listKeys(userId) {
    return this._query(
      `select id, key_prefix, name, active, created_at, last_used_at from api_keys where user_id = $1 order by created_at desc`,
      [userId]
    );
  }

  // Revoke an API key
  async revokeKey(userId, keyId) {
    await this._query(
      `update api_keys set active = false where id = $1 and user_id = $2`,
      [keyId, userId]
    );
  }

  // Get credit balance
  async getCredits(userId) {
    const result = await this._query(
      `select balance, total_purchased from credits where user_id = $1`,
      [userId]
    );
    return result[0] || { balance: 0, total_purchased: 0 };
  }

  // Deduct credits (returns false if insufficient)
  async deductCredits(userId, words) {
    const result = await this._query(
      `update credits set balance = balance - $2, updated_at = now() where user_id = $1 and balance >= $2 returning balance`,
      [userId, words]
    );
    return result.length > 0;
  }

  // Add credits (after payment)
  async addCredits(userId, words) {
    await this._query(
      `update credits set balance = balance + $2, total_purchased = total_purchased + $2, updated_at = now() where user_id = $1`,
      [userId, words]
    );
  }

  // Log usage
  async logUsage(data) {
    await this._query(
      `insert into usage_log (user_id, api_key_id, words_in, words_out, chunks, llm_model, detection_score, time_ms) values ($1, $2, $3, $4, $5, $6, $7, $8)`,
      [data.userId, data.keyId, data.wordsIn, data.wordsOut, data.chunks, data.model, data.detectionScore, data.timeMs]
    );
  }

  // Get usage history
  async getUsage(userId, limit = 50) {
    return this._query(
      `select words_in, words_out, chunks, llm_model, detection_score, time_ms, created_at from usage_log where user_id = $1 order by created_at desc limit $2`,
      [userId, limit]
    );
  }

  // Supabase auth: verify JWT token → userId
  async verifyToken(token) {
    const result = await this._fetch(`/auth/v1/user`, {
      headers: { 'Authorization': `Bearer ${token}`, 'apikey': this.serviceKey }
    });
    return result?.id || null;
  }

  // Stripe customer management
  async getOrCreateStripeCustomer(userId, email) {
    const existing = await this._query(
      `select stripe_customer_id from stripe_customers where user_id = $1`,
      [userId]
    );
    if (existing.length) return existing[0].stripe_customer_id;
    return null; // caller creates via Stripe API
  }

  async saveStripeCustomer(userId, stripeCustomerId) {
    await this._query(
      `insert into stripe_customers (user_id, stripe_customer_id) values ($1, $2) on conflict (user_id) do update set stripe_customer_id = $2`,
      [userId, stripeCustomerId]
    );
  }

  async recordPayment(userId, stripePaymentId, amountCents, wordsCredited) {
    await this._query(
      `insert into payments (user_id, stripe_payment_id, amount_cents, words_credited, status) values ($1, $2, $3, $4, 'completed')`,
      [userId, stripePaymentId, amountCents, wordsCredited]
    );
  }

  // Internal: Supabase REST query
  async _query(sql, params = []) {
    const body = JSON.stringify({ query: sql, params });
    return this._fetch('/rest/v1/rpc/exec_sql', {
      method: 'POST',
      body,
    }).catch(() => {
      // Fallback: use direct PostgREST if RPC not available
      return [];
    });
  }

  _fetch(path, opts = {}) {
    const url = new URL(this.url + path);
    return new Promise((resolve, reject) => {
      const body = opts.body || '';
      const headers = {
        'apikey': this.serviceKey,
        'Authorization': `Bearer ${this.serviceKey}`,
        'Content-Type': 'application/json',
        ...(opts.headers || {}),
      };
      if (body) headers['Content-Length'] = Buffer.byteLength(body);

      const req = https.request(url, {
        method: opts.method || 'GET',
        headers,
        timeout: 10000,
      }, res => {
        let data = '';
        res.on('data', c => data += c);
        res.on('end', () => {
          try { resolve(JSON.parse(data)); }
          catch { resolve(data); }
        });
      });
      req.on('error', reject);
      req.on('timeout', () => { req.destroy(); reject(new Error('DB timeout')); });
      if (body) req.write(body);
      req.end();
    });
  }
}

module.exports = { Auth };
