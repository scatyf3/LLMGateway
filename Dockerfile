FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY gateway ./gateway
RUN pip install --no-cache-dir .

COPY config ./config
ENV GATEWAY_CONFIG=/app/config/gateway.yaml

EXPOSE 8080
# One worker per container on purpose: all shared state (limits, budgets, caches)
# is in Redis, so scale out with more replicas. Per-process state (Prometheus
# registry, mock-backend "down" toggle) stays unambiguous.
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]
