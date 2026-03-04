const https = require('https');

// ============================================================
// IN-HOUSE AI DETECTOR
// Uses perplexity (token predictability) + burstiness (variance)
// No third-party detector API needed — just OpenAI logprobs
// ============================================================

const TOKENIZE_MODEL = 'gpt-4o-mini'; // cheap, fast, has logprobs

/**
 * Get per-token log probabilities from OpenAI
 * 
 * Approach: Ask the model to CONTINUE from a prefix of each sentence.
 * The logprobs on the continuation tell us how predictable the text is.
 * More predictable = lower perplexity = more likely AI-generated.
 * 
 * We split text into sentences, feed each as a completion prompt,
 * and measure how "surprised" the model is.
 */
function getLogprobsForSentence(sentence, apiKey, model = TOKENIZE_MODEL) {
  // Split sentence roughly in half — give first half as prompt, measure prediction of second half
  const words = sentence.split(/\s+/);
  if (words.length < 4) return Promise.resolve([]);

  const splitPoint = Math.floor(words.length / 2);
  const prefix = words.slice(0, splitPoint).join(' ');
  const expected = words.slice(splitPoint).join(' ');

  const body = JSON.stringify({
    model,
    messages: [
      {
        role: 'system',
        content: 'Continue the following text naturally. Write ONLY the next few words to complete the thought. Do not explain.'
      },
      { role: 'user', content: prefix }
    ],
    max_tokens: Math.min(100, expected.length),
    temperature: 0,
    logprobs: true,
    top_logprobs: 5,
  });

  return new Promise((resolve, reject) => {
    const url = new URL('https://api.openai.com/v1/chat/completions');
    const req = https.request(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
      timeout: 30000,
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
          const completion = parsed.choices?.[0]?.message?.content || '';

          // Measure similarity between expected continuation and actual completion
          // Higher similarity + high confidence (low logprobs) = AI-like text
          const expectedWords = expected.toLowerCase().split(/\s+/);
          const completionWords = completion.toLowerCase().split(/\s+/);

          let matchCount = 0;
          const checkLen = Math.min(expectedWords.length, completionWords.length);
          for (let i = 0; i < checkLen; i++) {
            // Fuzzy match — strip punctuation
            const e = expectedWords[i].replace(/[^a-z0-9]/g, '');
            const c = completionWords[i].replace(/[^a-z0-9]/g, '');
            if (e === c) matchCount++;
          }

          const predictability = checkLen > 0 ? matchCount / checkLen : 0;

          resolve({
            logprobs: content,
            predictability,
            avgLogprob: content.length > 0
              ? content.reduce((s, t) => s + (t.logprob || 0), 0) / content.length
              : -5,
          });
        } catch (e) {
          reject(new Error('Failed to parse logprobs response'));
        }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('Logprobs timeout')); });
    req.write(body);
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
 * Analyzes each sentence for predictability using OpenAI logprobs.
 * Returns: { score: 0-100, metrics, flagged_sentences }
 * score: 0 = AI, 100 = human
 */
async function detect(text, apiKey, options = {}) {
  const model = options.model || TOKENIZE_MODEL;
  const sentences = splitSentences(text);

  if (sentences.length === 0) {
    return { score: 50, error: 'No sentences found', details: {} };
  }

  const sentenceLengths = sentences.map(s => s.split(/\s+/).length);
  const lengthBurstiness = calcBurstiness(sentenceLengths);

  // Analyze a sample of sentences (max 15 to keep costs down)
  const sampleSize = Math.min(sentences.length, 15);
  const step = Math.max(1, Math.floor(sentences.length / sampleSize));
  const sampled = [];
  for (let i = 0; i < sentences.length && sampled.length < sampleSize; i += step) {
    if (sentences[i].split(/\s+/).length >= 4) {
      sampled.push({ text: sentences[i], index: i });
    }
  }

  // Get predictability for each sampled sentence
  const results = [];
  for (const s of sampled) {
    try {
      const r = await getLogprobsForSentence(s.text, apiKey, model);
      results.push({
        ...s,
        predictability: r.predictability,
        avgLogprob: r.avgLogprob,
        perplexity: Math.exp(-r.avgLogprob),
      });
    } catch (e) {
      // Skip failed sentences
      results.push({ ...s, predictability: 0.5, avgLogprob: -3, perplexity: 20, error: e.message });
    }
  }

  // Average predictability across sentences
  const avgPredictability = results.reduce((s, r) => s + r.predictability, 0) / results.length;
  const perplexities = results.map(r => r.perplexity);
  const avgPerplexity = perplexities.reduce((a, b) => a + b, 0) / perplexities.length;
  const perplexityBurstiness = calcPerplexityBurstiness(perplexities);

  // Predictability-based score:
  // High predictability (>0.6) = AI, Low predictability (<0.2) = human
  let predictScore;
  if (avgPredictability >= 0.6) predictScore = 0;
  else if (avgPredictability <= 0.15) predictScore = 100;
  else predictScore = ((0.6 - avgPredictability) / 0.45) * 100;

  // Length burstiness score
  let lengthScore;
  if (lengthBurstiness <= 0.1) lengthScore = 0;
  else if (lengthBurstiness >= 0.8) lengthScore = 100;
  else lengthScore = ((lengthBurstiness - 0.1) / 0.7) * 100;

  // Perplexity burstiness score
  let pBurstScore;
  if (perplexityBurstiness <= 0.15) pBurstScore = 0;
  else if (perplexityBurstiness >= 0.7) pBurstScore = 100;
  else pBurstScore = ((perplexityBurstiness - 0.15) / 0.55) * 100;

  // Weighted: predictability matters most
  const humanScore = Math.round(
    Math.max(0, Math.min(100,
      (predictScore * 0.5) + (lengthScore * 0.25) + (pBurstScore * 0.25)
    ))
  );

  // Flag highly predictable sentences
  const flaggedSentences = results
    .filter(r => r.predictability > 0.5)
    .map(r => ({
      text: r.text,
      predictability: Math.round(r.predictability * 1000) / 1000,
      perplexity: Math.round(r.perplexity * 100) / 100,
      word_count: r.text.split(/\s+/).length,
    }));

  return {
    score: humanScore,
    ai_probability: Math.round((100 - humanScore) / 100 * 1000) / 1000,
    human_probability: Math.round(humanScore / 100 * 1000) / 1000,
    metrics: {
      avg_predictability: Math.round(avgPredictability * 1000) / 1000,
      avg_perplexity: Math.round(avgPerplexity * 100) / 100,
      length_burstiness: Math.round(lengthBurstiness * 1000) / 1000,
      perplexity_burstiness: Math.round(perplexityBurstiness * 1000) / 1000,
    },
    sentence_count: sentences.length,
    sentences_analyzed: results.length,
    flagged_sentences: flaggedSentences,
    model_used: model,
  };
}

module.exports = { detect, splitSentences, calcPerplexity, calcBurstiness, computeScore };
