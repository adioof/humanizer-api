"""
Open-source AI text detector API
Uses desklib/ai-text-detector-v1.01 (DeBERTa-v3-large, #1 on RAID benchmark)
"""
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Model definition (from desklib) ---
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

# --- Load model ---
MODEL_NAME = "desklib/ai-text-detector-v1.01"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = DesklibAIDetectionModel.from_pretrained(MODEL_NAME)
device = torch.device("cpu")
model.to(device)
model.eval()
print("Model loaded.")

def predict(text, max_len=768, threshold=0.5):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    return {
        "ai_probability": round(probability, 4),
        "label": "AI" if probability >= threshold else "Human",
        "confidence": round(abs(probability - 0.5) * 2, 4)  # 0-1 scale
    }

def predict_chunked(text, chunk_size=512, overlap=64, threshold=0.5):
    """Split long texts into overlapping chunks, average the scores."""
    words = text.split()
    if len(words) <= chunk_size:
        return predict(text, threshold=threshold)

    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    scores = []
    for chunk in chunks:
        r = predict(chunk, threshold=threshold)
        scores.append(r["ai_probability"])

    avg_prob = sum(scores) / len(scores)
    return {
        "ai_probability": round(avg_prob, 4),
        "label": "AI" if avg_prob >= threshold else "Human",
        "confidence": round(abs(avg_prob - 0.5) * 2, 4),
        "chunks_analyzed": len(chunks),
        "chunk_scores": [round(s, 4) for s in scores]
    }

# --- Routes ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})

@app.route('/api/v1/detect', methods=['POST'])
def detect():
    body = request.get_json(silent=True) or {}
    text = body.get('text', '')
    threshold = body.get('threshold', 0.5)

    if not text or len(text.strip()) < 50:
        return jsonify({"error": "Text must be at least 50 characters"}), 400

    result = predict_chunked(text, threshold=threshold)
    result["model"] = MODEL_NAME
    result["model_type"] = "deberta-v3-large"
    result["benchmark"] = "RAID #1"
    return jsonify(result)

@app.route('/api/v1/batch', methods=['POST'])
def batch_detect():
    body = request.get_json(silent=True) or {}
    texts = body.get('texts', [])
    threshold = body.get('threshold', 0.5)

    if not texts:
        return jsonify({"error": "Provide a 'texts' array"}), 400

    results = []
    for text in texts[:50]:  # Max 50 texts per batch
        if len(text.strip()) < 50:
            results.append({"error": "Text too short", "text_preview": text[:50]})
        else:
            results.append(predict_chunked(text, threshold=threshold))

    return jsonify({"results": results, "model": MODEL_NAME})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
