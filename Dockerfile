FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV CHROMA_TELEMETRY_ENABLED=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Force clean DB to avoid schema mismatch
RUN rm -rf chroma_db

# Build vector DB inside container
RUN python ingest.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
