const https = require('https');

// ============================================================
// IN-HOUSE AI DETECTOR
// Uses perplexity (token predictability) + burstiness (variance)
// No third-party detector API needed — just OpenAI logprobs
// ============================================================

const TOKENIZE_MODEL = 'gpt-4o-mini'; // cheap, fast, has logprobs

/**
 * Get per-token log probabilities from OpenAI
 */
function getLogprobs(text, apiKey, model = TOKENIZE_MODEL) {
  const body = JSON.stringify({
    model,
    messages: [{ role: 'user', content: text }],
    max_tokens: 1, // we only care about the prompt logprobs
    logprobs: true,
    echo: false,
  });

  // Use completions endpoint for logprobs on input tokens
  // Chat completions only gives logprobs on OUTPUT tokens
  // So we use the old completions API with a newer model that supports it
  // Actually, let's use a different approach: ask the model to repeat the text
  // and capture logprobs on the output
  const repeatBody = JSON.stringify({
    model,
    messages: [
      {
        role: 'system',
        content: 'Repeat the following text EXACTLY as given. Do not change a single word, space, or punctuation mark. Output ONLY the repeated text.'
      },
      { role: 'user', content: text }
    ],
    max_tokens: Math.min(4096, Math.ceil(text.length / 2)),
    temperature: 0,
    logprobs: true,
    top_logprobs: 1,
  });

  return new Promise((resolve, reject) => {
    const url = new URL('https://api.openai.com/v1/chat/completions');
    const req = https.request(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(repeatBody),
      },
      timeout: 60000,
    }, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          reject(new Error(`OpenAI error ${res.statusCode}: ${data.slice(0, 300)}`));
          return;
        }
        try {
          const parsed = JSON.parse(data);
          const content = parsed.choices?.[0]?.logprobs?.content || [];
          resolve(content);
        } catch (e) {
          reject(new Error('Failed to parse logprobs response'));
        }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('Logprobs timeout')); });
    req.write(repeatBody);
    req.end();
  });
}

/**
 * Split text into sentences (simple but effective)
 */
function splitSentences(text) {
  // Split on sentence-ending punctuation followed by space or newline
  const raw = text.split(/(?<=[.!?])\s+|\n\n+/);
  return raw
    .map(s => s.trim())
    .filter(s => s.length > 10); // skip tiny fragments
}

/**
 * Calculate perplexity from logprobs
 * Perplexity = exp(-1/N * sum(log_probs))
 */
function calcPerplexity(logprobs) {
  if (!logprobs.length) return 0;
  const avgLogprob = logprobs.reduce((sum, t) => sum + (t.logprob || 0), 0) / logprobs.length;
  return Math.exp(-avgLogprob);
}

/**
 * Calculate burstiness (coefficient of variation of sentence lengths)
 */
function calcBurstiness(sentenceLengths) {
  if (sentenceLengths.length < 2) return 0;
  const mean = sentenceLengths.reduce((a, b) => a + b, 0) / sentenceLengths.length;
  if (mean === 0) return 0;
  const variance = sentenceLengths.reduce((sum, l) => sum + (l - mean) ** 2, 0) / sentenceLengths.length;
  const stdDev = Math.sqrt(variance);
  return stdDev / mean; // coefficient of variation
}

/**
 * Calculate burstiness of perplexity across sentences
 */
function calcPerplexityBurstiness(perplexities) {
  if (perplexities.length < 2) return 0;
  const mean = perplexities.reduce((a, b) => a + b, 0) / perplexities.length;
  if (mean === 0) return 0;
  const variance = perplexities.reduce((sum, p) => sum + (p - mean) ** 2, 0) / perplexities.length;
  return Math.sqrt(variance) / mean;
}

/**
 * Score text: 0 = definitely AI, 100 = definitely human
 * 
 * Thresholds calibrated from research:
 * - AI text: perplexity ~5-20, burstiness ~0.1-0.3
 * - Human text: perplexity ~30-80+, burstiness ~0.5-1.5+
 */
function computeScore(perplexity, lengthBurstiness, perplexityBurstiness) {
  // Perplexity score: higher = more human-like
  // Scale: 5 (AI) → 60+ (human)
  let perplexityScore;
  if (perplexity <= 5) perplexityScore = 0;
  else if (perplexity >= 60) perplexityScore = 100;
  else perplexityScore = ((perplexity - 5) / 55) * 100;

  // Length burstiness score: higher variance = more human
  // Scale: 0.1 (AI) → 0.8+ (human)
  let lengthScore;
  if (lengthBurstiness <= 0.1) lengthScore = 0;
  else if (lengthBurstiness >= 0.8) lengthScore = 100;
  else lengthScore = ((lengthBurstiness - 0.1) / 0.7) * 100;

  // Perplexity burstiness: higher = more human
  // Scale: 0.15 (AI) → 0.7+ (human)
  let pBurstScore;
  if (perplexityBurstiness <= 0.15) pBurstScore = 0;
  else if (perplexityBurstiness >= 0.7) pBurstScore = 100;
  else pBurstScore = ((perplexityBurstiness - 0.15) / 0.55) * 100;

  // Weighted combination: perplexity matters most
  const score = (perplexityScore * 0.5) + (lengthScore * 0.25) + (pBurstScore * 0.25);
  return Math.round(Math.max(0, Math.min(100, score)));
}

/**
 * Main detection function
 * Returns: { score: 0-100, perplexity, burstiness, details }
 * score: 0 = AI, 100 = human
 */
async function detect(text, apiKey, options = {}) {
  const model = options.model || TOKENIZE_MODEL;

  // Get logprobs for the full text
  const logprobs = await getLogprobs(text, apiKey, model);

  if (!logprobs.length) {
    return { score: 50, error: 'No logprobs returned', details: {} };
  }

  // Overall perplexity
  const overallPerplexity = calcPerplexity(logprobs);

  // Split into sentences for burstiness
  const sentences = splitSentences(text);
  const sentenceLengths = sentences.map(s => s.split(/\s+/).length);

  // Length burstiness
  const lengthBurstiness = calcBurstiness(sentenceLengths);

  // Per-sentence perplexity (approximate by splitting logprobs by sentence boundaries)
  // This is a rough approximation — we assign logprobs to sentences by token count
  const sentencePerplexities = [];
  let tokenIdx = 0;
  for (const sentence of sentences) {
    const approxTokens = Math.ceil(sentence.length / 4); // rough char-to-token ratio
    const sentLogprobs = logprobs.slice(tokenIdx, tokenIdx + approxTokens);
    if (sentLogprobs.length > 0) {
      sentencePerplexities.push(calcPerplexity(sentLogprobs));
    }
    tokenIdx += approxTokens;
  }

  const perplexityBurstiness = calcPerplexityBurstiness(sentencePerplexities);

  // Compute final score
  const humanScore = computeScore(overallPerplexity, lengthBurstiness, perplexityBurstiness);

  // Flag individual sentences that look AI-generated
  const flaggedSentences = [];
  for (let i = 0; i < sentences.length; i++) {
    const pplx = sentencePerplexities[i];
    if (pplx !== undefined && pplx < 15 && sentenceLengths[i] > 5) {
      flaggedSentences.push({
        text: sentences[i],
        perplexity: Math.round(pplx * 100) / 100,
        word_count: sentenceLengths[i],
      });
    }
  }

  return {
    score: humanScore, // 0 = AI, 100 = human
    ai_probability: Math.round((100 - humanScore) / 100 * 1000) / 1000,
    human_probability: Math.round(humanScore / 100 * 1000) / 1000,
    metrics: {
      perplexity: Math.round(overallPerplexity * 100) / 100,
      length_burstiness: Math.round(lengthBurstiness * 1000) / 1000,
      perplexity_burstiness: Math.round(perplexityBurstiness * 1000) / 1000,
    },
    sentence_count: sentences.length,
    flagged_sentences: flaggedSentences,
    model_used: model,
  };
}

module.exports = { detect, splitSentences, calcPerplexity, calcBurstiness, computeScore };
