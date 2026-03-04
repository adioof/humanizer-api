const https = require('https');

// ============================================================
// IN-HOUSE AI DETECTOR
// Uses perplexity (token predictability) + burstiness (variance)
// No third-party detector API needed — just OpenAI logprobs
// ============================================================

const TOKENIZE_MODEL = 'gpt-4o-mini'; // cheap, fast, has logprobs

/**
 * Get predictability of a sentence given its context.
 * 
 * Approach: Feed prior context + first half of sentence as a prompt.
 * Ask the model to continue. Measure word overlap with actual second half.
 * High overlap = model can predict this text = likely AI-generated.
 * 
 * Context is key — AI text is predictable IN CONTEXT, not in isolation.
 */
function predictSentence(sentence, priorContext, apiKey, model = TOKENIZE_MODEL) {
  const words = sentence.split(/\s+/);
  if (words.length < 4) return Promise.resolve({ predictability: 0, avgLogprob: -5 });

  // Give first 40% as prompt, predict the remaining 60%
  const splitPoint = Math.max(2, Math.floor(words.length * 0.4));
  const prefix = words.slice(0, splitPoint).join(' ');
  const expected = words.slice(splitPoint).join(' ');
  const expectedWordCount = words.length - splitPoint;

  // Include up to 200 words of prior context for better prediction
  const contextWords = priorContext.split(/\s+/).slice(-200).join(' ');
  const fullPrompt = contextWords ? `${contextWords} ${prefix}` : prefix;

  const body = JSON.stringify({
    model,
    messages: [
      {
        role: 'system',
        content: 'Continue the following text naturally. Write exactly the next ' + expectedWordCount + ' words. Do not explain, do not add anything extra.'
      },
      { role: 'user', content: fullPrompt }
    ],
    max_tokens: Math.min(150, expectedWordCount * 3),
    temperature: 0,
    logprobs: true,
    top_logprobs: 3,
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

          // Word-level overlap (case-insensitive, punctuation-stripped)
          const normalize = w => w.toLowerCase().replace(/[^a-z0-9]/g, '');
          const expectedArr = expected.split(/\s+/).map(normalize).filter(w => w.length > 0);
          const completionArr = completion.split(/\s+/).map(normalize).filter(w => w.length > 0);

          // Sliding window match — check if expected words appear in order in completion
          let matchCount = 0;
          let compIdx = 0;
          for (const ew of expectedArr) {
            // Look ahead up to 3 positions for a match (allows small insertions)
            for (let look = 0; look < 3 && compIdx + look < completionArr.length; look++) {
              if (completionArr[compIdx + look] === ew) {
                matchCount++;
                compIdx = compIdx + look + 1;
                break;
              }
            }
          }

          const predictability = expectedArr.length > 0 ? matchCount / expectedArr.length : 0;

          // Model confidence (how sure it was about its own output)
          const avgLogprob = content.length > 0
            ? content.reduce((s, t) => s + (t.logprob || 0), 0) / content.length
            : -5;

          // Confidence = how sure the model was about its own output
          const confidence = Math.exp(avgLogprob); // 0-1

          // Combined: predictability is the primary signal
          // Confidence only boosts when predictability is already moderate+
          // This prevents high-confidence-but-wrong completions from inflating scores
          const combinedScore = predictability >= 0.15
            ? predictability * 0.85 + confidence * 0.15
            : predictability;

          resolve({
            predictability,
            confidence,
            combinedScore,
            avgLogprob,
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

  // Get predictability for each sampled sentence WITH context
  const results = [];
  for (const s of sampled) {
    // Build prior context from all sentences before this one
    const priorContext = sentences.slice(0, s.index).join(' ');
    try {
      const r = await predictSentence(s.text, priorContext, apiKey, model);
      results.push({
        ...s,
        predictability: r.predictability,
        confidence: r.confidence,
        combinedScore: r.combinedScore,
        avgLogprob: r.avgLogprob,
      });
    } catch (e) {
      results.push({ ...s, predictability: 0.3, confidence: 0.5, combinedScore: 0.3, avgLogprob: -3, error: e.message });
    }
  }

  // Average scores across sentences
  const avgPredictability = results.reduce((s, r) => s + r.predictability, 0) / results.length;
  const avgCombined = results.reduce((s, r) => s + (r.combinedScore || 0), 0) / results.length;
  const avgConfidence = results.reduce((s, r) => s + (r.confidence || 0), 0) / results.length;
  const predictabilities = results.map(r => r.predictability);
  const predictBurstiness = calcPerplexityBurstiness(predictabilities); // reuse variance calc

  // Combined score:
  // High (>0.4) = AI, Low (<0.05) = human
  let predictScore;
  if (avgCombined >= 0.4) predictScore = 0;
  else if (avgCombined <= 0.05) predictScore = 100;
  else predictScore = ((0.4 - avgCombined) / 0.35) * 100;

  // Length burstiness score
  let lengthScore;
  if (lengthBurstiness <= 0.1) lengthScore = 0;
  else if (lengthBurstiness >= 0.8) lengthScore = 100;
  else lengthScore = ((lengthBurstiness - 0.1) / 0.7) * 100;

  // Predictability burstiness score (varied predictability = more human)
  let pBurstScore;
  if (predictBurstiness <= 0.15) pBurstScore = 0;
  else if (predictBurstiness >= 0.7) pBurstScore = 100;
  else pBurstScore = ((predictBurstiness - 0.15) / 0.55) * 100;

  // Weighted: predictability matters most
  const humanScore = Math.round(
    Math.max(0, Math.min(100,
      (predictScore * 0.5) + (lengthScore * 0.25) + (pBurstScore * 0.25)
    ))
  );

  // Flag highly predictable sentences (combined > 0.4)
  const flaggedSentences = results
    .filter(r => (r.combinedScore || 0) > 0.4)
    .map(r => ({
      text: r.text,
      predictability: Math.round(r.predictability * 1000) / 1000,
      confidence: Math.round((r.confidence || 0) * 1000) / 1000,
      combined: Math.round((r.combinedScore || 0) * 1000) / 1000,
      word_count: r.text.split(/\s+/).length,
    }));

  return {
    score: humanScore,
    ai_probability: Math.round((100 - humanScore) / 100 * 1000) / 1000,
    human_probability: Math.round(humanScore / 100 * 1000) / 1000,
    metrics: {
      avg_predictability: Math.round(avgPredictability * 1000) / 1000,
      avg_confidence: Math.round(avgConfidence * 1000) / 1000,
      avg_combined: Math.round(avgCombined * 1000) / 1000,
      length_burstiness: Math.round(lengthBurstiness * 1000) / 1000,
      predict_burstiness: Math.round(predictBurstiness * 1000) / 1000,
    },
    sentence_count: sentences.length,
    sentences_analyzed: results.length,
    flagged_sentences: flaggedSentences,
    model_used: model,
  };
}

module.exports = { detect, splitSentences, calcPerplexity, calcBurstiness, computeScore };
