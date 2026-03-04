const https = require('https');

// ============================================================
// IN-HOUSE AI DETECTOR v5
// Multi-signal approach:
// 1. Vocabulary richness (type-token ratio, hapax legomena)
// 2. Sentence length burstiness
// 3. Structural predictability (transition patterns)
// 4. OpenAI logprobs for spot-checking flagged sentences
// ============================================================

/**
 * Split text into sentences
 */
function splitSentences(text) {
  const raw = text.split(/(?<=[.!?])\s+|\n\n+/);
  return raw.map(s => s.trim()).filter(s => s.length > 10);
}

/**
 * Tokenize into words (lowercase, strip punctuation)
 */
function tokenize(text) {
  return text.toLowerCase()
    .replace(/[^a-z0-9\s'-]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 1);
}

/**
 * Signal 1: Vocabulary Richness
 * AI text uses more generic, common vocabulary.
 * Human text has more unique words, unusual word choices.
 */
function vocabRichness(words) {
  if (words.length < 20) return { ttr: 0.5, hapaxRatio: 0.5 };

  const unique = new Set(words);
  const ttr = unique.size / words.length; // Type-Token Ratio

  // Hapax legomena: words that appear exactly once
  const freq = {};
  for (const w of words) freq[w] = (freq[w] || 0) + 1;
  const hapax = Object.values(freq).filter(c => c === 1).length;
  const hapaxRatio = hapax / unique.size;

  return { ttr: Math.round(ttr * 1000) / 1000, hapaxRatio: Math.round(hapaxRatio * 1000) / 1000 };
}

/**
 * Signal 2: Sentence Length Burstiness
 * AI = uniform sentence lengths, Human = varied
 */
function calcBurstiness(values) {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (mean === 0) return 0;
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
  return Math.round((Math.sqrt(variance) / mean) * 1000) / 1000;
}

/**
 * Signal 3: AI-typical patterns
 * Count transition phrases, filler phrases, and structural patterns common in AI text
 */
const AI_PATTERNS = [
  /\bin (?:the|today's) (?:rapidly |ever-)?(?:evolving|changing) (?:landscape|world)/i,
  /\blet's (?:dive|delve|explore|unpack)/i,
  /\bit'?s worth noting/i,
  /\bin (?:this|today's) (?:article|post|guide)/i,
  /\bgame[- ]changer/i,
  /\blever(?:age|aging)/i,
  /\bcomprehensive (?:guide|overview|look)/i,
  /\bstream ?line/i,
  /\brobust (?:solution|framework|system|platform)/i,
  /\bcutting[- ]edge/i,
  /\bunprecedented (?:opportunity|growth|level)/i,
  /\btransformative/i,
  /\bseamless(?:ly)? integrat/i,
  /\bholistic approach/i,
  /\bparadigm shift/i,
  /\bin conclusion/i,
  /\bfurthermore,/i,
  /\bmoreover,/i,
  /\bconsequently,/i,
  /\bnevertheless,/i,
  /\bsignificant(?:ly)? (?:enhance|improve|boost|impact)/i,
  /\bundoubtedly/i,
  /\bempowe?r(?:ing|s)? (?:developer|team|user|organization)/i,
  /\bfoster(?:ing|s)? (?:innovation|collaboration|growth)/i,
  /\bnavigate (?:the|this) (?:complex|changing)/i,
];

function countAIPatterns(text) {
  let count = 0;
  for (const pattern of AI_PATTERNS) {
    if (pattern.test(text)) count++;
  }
  return count;
}

/**
 * Signal 4: Sentence start diversity
 * AI tends to start sentences with similar structures
 */
function sentenceStartDiversity(sentences) {
  if (sentences.length < 3) return 1;
  const starts = sentences.map(s => {
    const words = s.split(/\s+/).slice(0, 2).join(' ').toLowerCase();
    return words;
  });
  const unique = new Set(starts);
  return Math.round((unique.size / starts.length) * 1000) / 1000;
}

/**
 * Signal 5: Paragraph length variation
 */
function paragraphVariation(text) {
  const paragraphs = text.split(/\n\n+/).filter(p => p.trim().length > 0);
  if (paragraphs.length < 2) return 0.5;
  const lengths = paragraphs.map(p => p.split(/\s+/).length);
  return calcBurstiness(lengths);
}

/**
 * Optional: OpenAI spot-check for flagged sentences
 */
function spotCheck(sentence, priorContext, apiKey, model = 'gpt-4o-mini') {
  const words = sentence.split(/\s+/);
  if (words.length < 5) return Promise.resolve({ predictability: 0 });

  const splitPoint = Math.max(2, Math.floor(words.length * 0.4));
  const prefix = words.slice(0, splitPoint).join(' ');
  const expected = words.slice(splitPoint);
  const contextWords = priorContext.split(/\s+/).slice(-150).join(' ');
  const fullPrompt = contextWords ? `${contextWords} ${prefix}` : prefix;

  const body = JSON.stringify({
    model,
    messages: [
      { role: 'system', content: 'Continue this text naturally. Write the next ' + expected.length + ' words only.' },
      { role: 'user', content: fullPrompt }
    ],
    max_tokens: Math.min(100, expected.length * 3),
    temperature: 0,
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
      timeout: 20000,
    }, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode >= 400) { reject(new Error(`OpenAI ${res.statusCode}`)); return; }
        try {
          const parsed = JSON.parse(data);
          const completion = parsed.choices?.[0]?.message?.content || '';
          const normalize = w => w.toLowerCase().replace(/[^a-z0-9]/g, '');
          const expArr = expected.map(normalize).filter(w => w);
          const compArr = completion.split(/\s+/).map(normalize).filter(w => w);
          let match = 0, ci = 0;
          for (const ew of expArr) {
            for (let look = 0; look < 3 && ci + look < compArr.length; look++) {
              if (compArr[ci + look] === ew) { match++; ci = ci + look + 1; break; }
            }
          }
          resolve({ predictability: expArr.length > 0 ? match / expArr.length : 0 });
        } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });
    req.write(body);
    req.end();
  });
}

/**
 * Main detection function
 * Returns: { score: 0-100 (0=AI, 100=human), metrics, flagged_sentences }
 */
async function detect(text, apiKey, options = {}) {
  const model = options.model || 'gpt-4o-mini';
  const sentences = splitSentences(text);
  const words = tokenize(text);

  if (words.length < 20) {
    return { score: 50, error: 'Text too short for reliable detection' };
  }

  // ── Signal 1: Vocabulary ──
  const vocab = vocabRichness(words);
  // AI: TTR ~0.4-0.55, Human: TTR ~0.55-0.75+
  let vocabScore;
  if (vocab.ttr <= 0.4) vocabScore = 0;
  else if (vocab.ttr >= 0.7) vocabScore = 100;
  else vocabScore = ((vocab.ttr - 0.4) / 0.3) * 100;

  // ── Signal 2: Sentence length burstiness ──
  const sentLengths = sentences.map(s => s.split(/\s+/).length);
  const lengthBurst = calcBurstiness(sentLengths);
  // AI: ~0.15-0.3, Human: ~0.4-1.0+
  let burstScore;
  if (lengthBurst <= 0.15) burstScore = 0;
  else if (lengthBurst >= 0.8) burstScore = 100;
  else burstScore = ((lengthBurst - 0.15) / 0.65) * 100;

  // ── Signal 3: AI pattern count ──
  const aiPatterns = countAIPatterns(text);
  // 0 patterns = human, 3+ = definitely AI
  let patternScore;
  if (aiPatterns >= 3) patternScore = 0;
  else if (aiPatterns === 0) patternScore = 100;
  else patternScore = (1 - aiPatterns / 3) * 100;

  // ── Signal 4: Sentence start diversity ──
  const startDiv = sentenceStartDiversity(sentences);
  // AI: ~0.5-0.7, Human: ~0.8-1.0
  let divScore;
  if (startDiv <= 0.5) divScore = 0;
  else if (startDiv >= 0.95) divScore = 100;
  else divScore = ((startDiv - 0.5) / 0.45) * 100;

  // ── Signal 5: Paragraph variation ──
  const paraVar = paragraphVariation(text);
  let paraScore;
  if (paraVar <= 0.1) paraScore = 0;
  else if (paraVar >= 0.7) paraScore = 100;
  else paraScore = ((paraVar - 0.1) / 0.6) * 100;

  // ── Weighted combination ──
  const humanScore = Math.round(Math.max(0, Math.min(100,
    (vocabScore * 0.2) +
    (burstScore * 0.2) +
    (patternScore * 0.3) +   // AI patterns are the strongest signal
    (divScore * 0.15) +
    (paraScore * 0.15)
  )));

  // ── Spot-check suspicious sentences with OpenAI ──
  const flaggedSentences = [];

  if (apiKey && humanScore < 75) {
    // Only spot-check if the text looks somewhat suspicious
    const sampled = sentences.slice(0, Math.min(sentences.length, 8));
    for (let i = 0; i < sampled.length; i++) {
      if (sampled[i].split(/\s+/).length < 5) continue;
      try {
        const priorContext = sentences.slice(0, i).join(' ');
        const result = await spotCheck(sampled[i], priorContext, apiKey, model);
        if (result.predictability > 0.4) {
          flaggedSentences.push({
            text: sampled[i],
            predictability: Math.round(result.predictability * 1000) / 1000,
            word_count: sampled[i].split(/\s+/).length,
          });
        }
      } catch (e) {
        // Skip failed spot checks
      }
    }
  }

  // Adjust score based on spot-check results
  let adjustedScore = humanScore;
  if (flaggedSentences.length > 0) {
    const flaggedRatio = flaggedSentences.length / Math.min(sentences.length, 8);
    adjustedScore = Math.round(humanScore * (1 - flaggedRatio * 0.3));
  }

  return {
    score: adjustedScore,
    ai_probability: Math.round((100 - adjustedScore) / 100 * 1000) / 1000,
    human_probability: Math.round(adjustedScore / 100 * 1000) / 1000,
    metrics: {
      vocab_ttr: vocab.ttr,
      hapax_ratio: vocab.hapaxRatio,
      length_burstiness: lengthBurst,
      ai_patterns_found: aiPatterns,
      sentence_start_diversity: startDiv,
      paragraph_variation: paraVar,
    },
    sub_scores: {
      vocabulary: Math.round(vocabScore),
      burstiness: Math.round(burstScore),
      ai_patterns: Math.round(patternScore),
      start_diversity: Math.round(divScore),
      paragraph_variation: Math.round(paraScore),
    },
    sentence_count: sentences.length,
    word_count: words.length,
    flagged_sentences: flaggedSentences,
    model_used: model,
  };
}

module.exports = { detect, splitSentences, tokenize, vocabRichness, calcBurstiness, countAIPatterns };
