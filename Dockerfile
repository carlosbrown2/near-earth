FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir . ".[api]"

# Copy application code
COPY prospector/ prospector/
COPY scoring/ scoring/

# Create data directory mount point
RUN mkdir -p /data

ENV PROSPECTOR_DB_PATH=/data/prospector.db
ENV PROSPECTOR_API_KEYS=""

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "prospector.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
