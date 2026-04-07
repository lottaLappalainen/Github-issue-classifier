FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model
COPY src/ ./src/
COPY models/ ./models/
COPY metrics.json ./metrics.json

EXPOSE 8000

CMD ["uvicorn", "src.api.serve:app", "--host", "0.0.0.0", "--port", "8000"]