const https = require('https');
const querystring = require('querystring');

// ============================================================
// STRIPE INTEGRATION
// ============================================================

// Pricing tiers (words per dollar)
const PLANS = {
  starter: { words: 10000, price_cents: 500, name: 'Starter — 10K words' },
  pro: { words: 50000, price_cents: 1900, name: 'Pro — 50K words' },
  business: { words: 200000, price_cents: 4900, name: 'Business — 200K words' },
};

class StripeClient {
  constructor(secretKey) {
    this.secretKey = secretKey;
  }

  // Create a customer
  async createCustomer(email) {
    const data = await this._request('POST', '/v1/customers', { email });
    return data.id;
  }

  // Create a checkout session for buying credits
  async createCheckoutSession(customerId, planKey, successUrl, cancelUrl) {
    const plan = PLANS[planKey];
    if (!plan) throw new Error(`Unknown plan: ${planKey}. Options: ${Object.keys(PLANS).join(', ')}`);

    const data = await this._request('POST', '/v1/checkout/sessions', {
      customer: customerId,
      mode: 'payment',
      'line_items[0][price_data][currency]': 'usd',
      'line_items[0][price_data][product_data][name]': plan.name,
      'line_items[0][price_data][unit_amount]': plan.price_cents,
      'line_items[0][quantity]': 1,
      success_url: successUrl || 'https://aucto.ai?payment=success',
      cancel_url: cancelUrl || 'https://aucto.ai?payment=cancel',
      'metadata[plan]': planKey,
      'metadata[words]': plan.words,
    });

    return { url: data.url, sessionId: data.id };
  }

  // Verify webhook signature
  verifyWebhook(payload, signature, secret) {
    const elements = signature.split(',');
    const timestamp = elements.find(e => e.startsWith('t='))?.slice(2);
    const v1 = elements.find(e => e.startsWith('v1='))?.slice(3);

    if (!timestamp || !v1) return false;

    const crypto = require('crypto');
    const expected = crypto.createHmac('sha256', secret)
      .update(`${timestamp}.${payload}`)
      .digest('hex');

    return crypto.timingSafeEqual(Buffer.from(v1), Buffer.from(expected));
  }

  // Parse checkout.session.completed event
  parseCheckoutEvent(event) {
    if (event.type !== 'checkout.session.completed') return null;
    const session = event.data.object;
    return {
      customerId: session.customer,
      paymentId: session.payment_intent,
      amountCents: session.amount_total,
      plan: session.metadata?.plan,
      words: parseInt(session.metadata?.words) || 0,
    };
  }

  _request(method, path, params = {}) {
    const body = querystring.stringify(params);
    return new Promise((resolve, reject) => {
      const req = https.request({
        hostname: 'api.stripe.com',
        path,
        method,
        headers: {
          'Authorization': `Basic ${Buffer.from(this.secretKey + ':').toString('base64')}`,
          'Content-Type': 'application/x-www-form-urlencoded',
          'Content-Length': Buffer.byteLength(body),
        },
        timeout: 15000,
      }, res => {
        let data = '';
        res.on('data', c => data += c);
        res.on('end', () => {
          try {
            const parsed = JSON.parse(data);
            if (parsed.error) reject(new Error(parsed.error.message));
            else resolve(parsed);
          } catch { reject(new Error('Stripe parse error')); }
        });
      });
      req.on('error', reject);
      req.on('timeout', () => { req.destroy(); reject(new Error('Stripe timeout')); });
      req.write(body);
      req.end();
    });
  }
}

module.exports = { StripeClient, PLANS };
