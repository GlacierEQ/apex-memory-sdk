FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir cryptography numpy coremltools torch

COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./

RUN pip install --no-cache-dir -e .

ENV APEX_MEMORY_DB=/data/memory.sqlite3
RUN mkdir -p /data

EXPOSE 18421

CMD ["apex-memory", "serve", "--port", "18421", "--db-path", "/data/memory.sqlite3"]
