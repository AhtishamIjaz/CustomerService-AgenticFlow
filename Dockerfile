# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

# Install uv for high-speed installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# COPY YOUR ACTUAL MANIFEST FILE
COPY requirements.txt ./ 

# Install dependencies into a virtual environment
RUN uv venv /app/.venv && \
    uv pip install --no-cache -r requirements.txt

# Stage 2: Final Production Image
FROM python:3.11-slim

# Install FAISS and Healthcheck dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the environment and code
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Environment Setup
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]