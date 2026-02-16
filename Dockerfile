FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY darwin_agent/ darwin_agent/
COPY config_example.yaml .

# Data dirs
RUN mkdir -p data/generations data/logs

EXPOSE 8080

# Default: paper trading
CMD ["python", "-m", "darwin_agent", "--mode", "test"]
