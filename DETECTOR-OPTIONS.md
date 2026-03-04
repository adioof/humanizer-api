# Open-Source AI Detector — Options

## Best Models Available
1. **desklib/ai-text-detector-v1.01** — DeBERTa-v3-large, #1 on RAID benchmark. 1.7GB model.
2. **fakespot-ai/roberta-base-ai-text-detection-v1** — RoBERTa-base, 500MB. Lighter.
3. **SuperAnnotate/roberta-large-llm-content-detector** — RoBERTa-large.

## Deployment Options

### Option A: Railway (always-on) — $20-30/mo
- Deploy Python + PyTorch service
- 2-3GB RAM needed = $20-30/mo on Railway
- Pro: fast, reliable, always available
- Con: expensive for side project

### Option B: HuggingFace Inference API — FREE (with token)
- Sign up at huggingface.co, get free API token
- Free tier: rate-limited, CPU-only, cold starts
- Pro: zero cost, zero ops
- Con: cold starts (30s+), rate limits, may go down

### Option C: HuggingFace Serverless Inference Endpoint — ~$5-10/mo
- Pay per second of compute
- Auto-scales to zero
- Pro: only pay when used
- Con: still has cold start on scale-up

### Option D: Modal / Replicate / Banana — Pay per inference
- Serverless GPU platforms
- ~$0.001-0.005 per inference call
- Pro: no idle cost, fast GPU inference
- Con: need account setup

### Option E: Self-host on humanizer-api (Node.js + ONNX) — $0 extra
- Convert model to ONNX format
- Run with onnxruntime-node in existing service
- Pro: no extra cost, no extra service
- Con: need to convert model, adds ~1GB to Docker image, inference slower on CPU

## Recommendation
**Option E** for now — add ONNX model to humanizer-api.
**Option B** as fallback — just need a HF token (free signup).
**Option A** if we need production reliability later.
