# AI Detection Benchmark Analysis

## GPTZero's Published Benchmarks (v4.1b, 2026)

### Standard Detection (no humanization)
| Domain | FPR | Recall | Precision | Accuracy |
|--------|-----|--------|-----------|----------|
| Academic Papers | 0.00% | 99.70% | 100.00% | 99.85% |
| Creative Writing | 0.00% | 97.40% | 100.00% | 98.70% |
| Essays | 0.00% | 99.70% | 100.00% | 99.85% |
| Product Reviews | 0.00% | 98.30% | 100.00% | 99.15% |
| **Average** | **0.00%** | **98.78%** | **100.00%** | **99.39%** |

### Key Takeaways
- GPTZero catches 98.78% of AI text with 0% false positives
- They test against GPT-5.2, Gemini 3 Pro, Claude Sonnet 4.5, Grok 4 Fast
- They specifically train against humanization/bypassing techniques
- 15 model releases in 2025 alone — they actively counter new bypass methods
- They classify: Human | Mixed (Polished) | Mixed (Concatenated) | AI (Pure) | AI (Paraphrased)

### What Works Against GPTZero (from research)
- Undetectable AI: 10-46% scores on GPTZero (free tier)
- GPTHumanizer AI: "reliably bypasses" (Jan 2026)
- WriteHybrid: claims 99%+ success rate
- BypassGPT: "near-perfect human scores" after humanization
- Phrasly: COMPLETE FAILURE on GPTZero

### What GPTZero Measures
1. **Perplexity**: Token-level predictability via fine-tuned RoBERTa model
2. **Burstiness**: Sentence structure variation
3. **Token probability patterns**: Clustering around high-probability choices
4. **Multi-class classification**: Trained on human, AI, polished, paraphrased datasets
5. **Adversarial robustness**: Specifically trained against humanizer outputs

## Our Current Position

### Our Detector (v6)
| Text Type | Our Score | GPTZero Result |
|-----------|-----------|----------------|
| Pure AI | 22/100 | 100% AI |
| Human (LinkedIn) | 82/100 | 98% Mixed (6/16 flagged) |
| Humanized (v1) | 79/100 | 100% AI |

### Gap Analysis
- **Detection accuracy**: We roughly agree on pure AI (both say AI). We disagree on humanized text (we say 79% human, GPTZero says 100% AI). Our detector is ~40 points too lenient.
- **Humanizer effectiveness**: Our humanized output is completely transparent to GPTZero. Not even "Mixed" — straight 100% AI.

## What We Need to Match Frontier

### Option A: Use GPTZero as Ground Truth ($45/mo)
- API: 300K words/month
- Use as the verification oracle in our humanize loop
- Our detector = fast pre-filter, GPTZero = final gatekeeper
- Cost: ~$0.015 per article check

### Option B: Fine-tune Our Own Classifier
- Need: thousands of human + AI text pairs across domains
- Model: Fine-tune RoBERTa-base on this dataset
- Hosting: Need GPU (Railway doesn't have GPUs cheaply)
- Timeline: weeks of work
- Advantage: no per-query cost, full control

### Option C: Hybrid (Recommended)
- Use GPTZero API as ground truth for the humanize loop
- Collect (text, GPTZero_score) pairs over time
- Once we have enough data, fine-tune our own classifier
- Gradually reduce dependence on GPTZero
- Our LLM-based detector stays as a fast pre-filter (no API cost)

## Humanizer Improvements Needed
Current approach (single LLM rewrite) is not enough. Need:
1. Multi-pass pipeline (structure → vocabulary → style)
2. Verification against actual GPTZero scores
3. Targeted rewriting of flagged sentences
4. Learning from successful bypasses over time
