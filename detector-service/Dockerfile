FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time (cached in Docker layer)
RUN python -c "
from transformers import AutoTokenizer, AutoModel, AutoConfig
print('Downloading desklib/ai-text-detector-v1.01...')
AutoTokenizer.from_pretrained('desklib/ai-text-detector-v1.01')
AutoConfig.from_pretrained('desklib/ai-text-detector-v1.01')
AutoModel.from_pretrained('desklib/ai-text-detector-v1.01')
print('Done.')
"

COPY server.py .

EXPOSE 8080
CMD ["python", "server.py"]
