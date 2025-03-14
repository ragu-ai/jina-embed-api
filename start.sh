#!/bin/bash


echo "installing requirements"
pip install -r requirements-cuda.txt

# Set the number of workers
# G5.xlarge has 4 vCPUs, so 2-3 workers is appropriate
NUM_WORKERS=${NUM_WORKERS:-2}

# Run uvicorn with optimized settings
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers $NUM_WORKERS \
  --log-level info \
  --limit-concurrency 20 \
  --timeout-keep-alive 120 \
  --backlog 1024 \