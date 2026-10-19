FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    DEEPEVAL_UPDATE_WARNING_OPT_IN=NO
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# The Copilot SDK drives a separate runtime binary it would otherwise download on first use (every
# `docker compose run --rm`). Bake it into the image instead.
RUN python -m copilot download-runtime

# docker-compose bind-mounts ./mcp-evaluator over /app so you can edit goldens/metrics without rebuilding.
COPY . .

ENTRYPOINT ["python", "run_evals.py"]
