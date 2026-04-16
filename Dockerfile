FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch first with explicit version >= 2.6
RUN pip install --no-cache-dir \
    "numpy<2" \
    "torch>=2.6.0" \
    "torchaudio>=2.6.0" \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python packages
RUN pip install --no-cache-dir \
    transformers \
    librosa \
    dtw-python \
    phonemizer \
    fastapi \
    python-multipart \
    pydub \
    gTTS \
    fastdtw \
    scipy \
    levenshtein \
    scikit-learn \
    streamlit \
    plotly \
    requests \
    uvicorn \
    soundfile \
    huggingface-hub

ENV HF_HOME=/app/.cache/huggingface
ENV TF_CPP_MIN_LOG_LEVEL=3

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]