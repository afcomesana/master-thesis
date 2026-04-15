ARG PYTHON_VERSION=3.13.12
ARG WEB_PORT=8080
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install su-exec to drop privileges in the entrypoint
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    && rm -rf /var/lib/apt/lists/*

RUN chown -R appuser:appuser /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && rm /app/entrypoint.sh

# Expose Django's port
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]

# Run Django with Uvicorn
CMD ["uvicorn", "thesis.asgi:application", "--host", "0.0.0.0", "--port", "8080", "--reload", "--reload-include", "'*.html'"]