# veris context engine — build from the REPO ROOT:
#   docker build -f context_engine/Dockerfile -t veris-context-engine .
#   docker run --rm -p 8090:8090 --env-file .env veris-context-engine
#
# The service only needs fastapi + uvicorn (all HTTP is stdlib urllib), so the
# image stays small and independent of the heavyweight root pyproject (torch etc.).

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29"

COPY context_engine /app/context_engine

# Run as non-root (cloud hosts care)
RUN useradd -m appuser
USER appuser

EXPOSE 8090

# Keys/knobs come from the environment (--env-file .env or the host's secret
# manager): OPENAI_API_KEY, GOOGLE_MAPS_API_KEY, CTX_* (see composer.py/maps.py).
CMD ["uvicorn", "context_engine.service:app", "--host", "0.0.0.0", "--port", "8090"]
