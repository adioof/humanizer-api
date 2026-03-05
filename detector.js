const https = require('https');

// ============================================================
// AI DETECTOR v7
// Uses HuggingFace open-source RoBERTa classifier (fakespot-ai)
// + statistical signals as secondary confirmation
// 
// Primary: fakespot-ai/roberta-base-ai-text-detection-v1 via HF Inference API
// Secondary: burstiness, vocabulary, AI pattern regex
// ============================================================

const HF_MODEL = 'fakespot-ai/roberta-base-ai-text-detection-v1';
const HF_API_URL = 'https://router.huggingface.co/hf-inference/models/' + HF_MODEL;

// ── Statistical helpers ──

function splitSentences(text) {
  return text.split(/(?<=[.!?])\s+|\n\n+/).map(s => s.trim()).filter(s => s.length > 10);
}

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s'-]/g, ' ').split(/\s+/).filter(w => w.length > 1);
}

function calcBurstiness(values) {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (mean === 0) return 0;
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
  return Math.round((Math.sqrt(variance) / mean) * 1000) / 1000;
}

function vocabRichness(words) {
  if (words.length < 20) return { ttr: 0.5 };
  const unique = new Set(words);
  return { ttr: Math.round((unique.size / words.length) * 1000) / 1000 };
}

const AI_PATTERNS = [
  /\bin (?:the|today's) (?:rapidly |ever-)?(?:evolving|changing) (?:landscape|world)/i,
  /\blet's (?:dive|delve|explore|unpack)/i,
  /\bit'?s worth noting/i,
  /\bgame[- ]changer/i,
  /\blever(?:age|aging)/i,
  /\bcomprehensive (?:guide|overview|look)/i,
  /\bstream ?line/i,
  /\brobust (?:solution|framework|system|platform)/i,
  /\bcutting[- ]edge/i,
  /\bunprecedented/i,
  /\btransformative/i,
  /\bseamless(?:ly)?/i,
  /\bholistic approach/i,
  /\bin conclusion/i,
  /\bfurthermore,/i,
  /\bmoreover,/i,
  /\bconsequently,/i,
  /\bnevertheless,/i,
  /\bundoubtedly/i,
  /\bempowe?r(?:ing|s)?/i,
  /\bfoster(?:ing|s)? (?:innovation|collaboration|growth)/i,
  /\bnavigate (?:the|this)/i,
  /\bparadigm shift/i,
  /\bin today's (?:world|age|era|digital)/i,
  /\bkey takeaway/i,
];

function countAIPatterns(text) {
  return AI_PATTERNS.reduce((c, p) => c + (p.test(text) ? 1 : 0), 0);
}

// ── HuggingFace Inference API ──

function hfClassify(text, hfToken) {
  // Truncate to ~2000 chars for API limits
  const truncated = text.length > 5000 ? text.substring(0, 5000) : text;

  const body = JSON.stringify({ inputs: truncated });

  return new Promise((resolve, reject) => {
    const url = new URL(HF_API_URL);
    const req = https.request(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${hfToken}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
      timeout: 30000,
    }, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          reject(new Error(`HuggingFace ${res.statusCode}: ${data.slice(0, 300)}`));
          return;
        }
        try {
          const parsed = JSON.parse(data);
          // HF returns [[{label, score}, ...]]  or [{label, score}, ...]
          const results = Array.isArray(parsed[0]) ? parsed[0] : parsed;
          const aiScore = results.find(r => r.label === 'AI' || r.label === 'LABEL_1')?.score || 0;
          const humanScore = results.find(r => r.label === 'Human' || r.label === 'LABEL_0')?.score || 0;
          resolve({ ai_probability: aiScore, human_probability: humanScore, raw: results });
        } catch (e) {
          reject(new Error('Failed to parse HF response: ' + e.message));
        }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('HF API timeout')); });
    req.write(body);
    req.end();
  });
}

/**
 * Chunked detection for longer texts
 * Split into chunks, classify each, return weighted average
 */
async function hfClassifyChunked(text, hfToken) {
  const words = text.split(/\s+/);

  // Short text — single call
  if (words.length <= 500) {
    return hfClassify(text, hfToken);
  }

  // Split into ~400 word chunks with 50 word overlap
  const chunkSize = 400;
  const overlap = 50;
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    chunks.push(words.slice(i, i + chunkSize).join(' '));
    i += chunkSize - overlap;
  }

  const results = [];
  for (const chunk of chunks) {
    try {
      const r = await hfClassify(chunk, hfToken);
      results.push(r);
    } catch (e) {
      // Skip failed chunks
    }
  }

  if (results.length === 0) {
    throw new Error('All chunks failed classification');
  }

  const avgAI = results.reduce((s, r) => s + r.ai_probability, 0) / results.length;
  const avgHuman = results.reduce((s, r) => s + r.human_probability, 0) / results.length;

  return {
    ai_probability: Math.round(avgAI * 10000) / 10000,
    human_probability: Math.round(avgHuman * 10000) / 10000,
    chunks_analyzed: results.length,
    chunk_scores: results.map(r => ({ ai: Math.round(r.ai_probability * 1000) / 1000, human: Math.round(r.human_probability * 1000) / 1000 })),
  };
}

/**
 * Main detection function
 * Primary: HuggingFace RoBERTa classifier (80% weight)
 * Secondary: Statistical signals (20% weight)
 */
async function detect(text, apiKeyOrHfToken, options = {}) {
  const sentences = splitSentences(text);
  const words = tokenize(text);

  if (words.length < 20) {
    return { score: 50, error: 'Text too short for reliable detection', provider: 'inhouse' };
  }

  // ── Statistical signals (instant, free) ──
  const sentLengths = sentences.map(s => s.split(/\s+/).length);
  const lengthBurst = calcBurstiness(sentLengths);
  const vocab = vocabRichness(words);
  const aiPatterns = countAIPatterns(text);

  let statScore = 50;
  if (aiPatterns >= 3) statScore -= 30;
  else if (aiPatterns >= 1) statScore -= 10 * aiPatterns;
  if (lengthBurst >= 0.5) statScore += 15;
  else if (lengthBurst <= 0.2) statScore -= 15;
  if (vocab.ttr >= 0.65) statScore += 10;
  else if (vocab.ttr <= 0.45) statScore -= 10;
  statScore = Math.max(0, Math.min(100, statScore));

  // ── HuggingFace classifier (primary) ──
  const hfToken = options.hf_token || apiKeyOrHfToken;
  let hfResult = null;
  let hfScore = null;

  try {
    hfResult = await hfClassifyChunked(text, hfToken);
    // Convert to 0-100 human score (matching our convention: 100 = human, 0 = AI)
    hfScore = Math.round(hfResult.human_probability * 100);
  } catch (e) {
    hfResult = { error: e.message };
  }

  // ── Combine: HF 80%, Stats 20% ──
  const finalScore = hfScore !== null
    ? Math.round(hfScore * 0.8 + statScore * 0.2)
    : statScore;

  return {
    score: Math.max(0, Math.min(100, finalScore)),
    label: finalScore >= 50 ? 'Human' : 'AI',
    ai_probability: Math.round((100 - finalScore) / 100 * 1000) / 1000,
    human_probability: Math.round(finalScore / 100 * 1000) / 1000,
    hf_score: hfScore,
    hf_ai_probability: hfResult?.ai_probability ?? null,
    hf_human_probability: hfResult?.human_probability ?? null,
    stat_score: statScore,
    metrics: {
      vocab_ttr: vocab.ttr,
      length_burstiness: lengthBurst,
      ai_patterns_found: aiPatterns,
      sentence_count: sentences.length,
      word_count: words.length,
    },
    chunks: hfResult?.chunks_analyzed || 1,
    chunk_scores: hfResult?.chunk_scores || null,
    model: HF_MODEL,
    provider: 'huggingface',
  };
}

module.exports = { detect, splitSentences, tokenize, vocabRichness, calcBurstiness, countAIPatterns };
