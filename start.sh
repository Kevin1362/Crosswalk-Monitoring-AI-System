#!/bin/sh

echo "======================================"
echo "Starting GuideKaro MLOps Application"
echo "======================================"

echo "Starting FastAPI on port 8000..."

uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 &

API_PID=$!

sleep 3

if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "ERROR: GuideKaro FastAPI failed to start."
    exit 1
fi

echo "FastAPI started successfully."

echo "Starting GuideKaro Streamlit Dashboard on port 8501..."

exec python -m streamlit run dashboard_latest.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true