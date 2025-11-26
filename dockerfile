# ----------------------------
# 1. Base Image
# ----------------------------
FROM python:3.11-slim

# Prevent prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------
# 2. Set working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# 3. Install system dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 4. Install Python deps first (layer caching)
# ----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# 5. Copy all project files
# ----------------------------
COPY . .

# ----------------------------
# 6. Build ChromaDB at build time
# ----------------------------
RUN python ingest.py

# ----------------------------
# 7. Expose port
# ----------------------------
EXPOSE 8000

# ----------------------------
# 8. Start FastAPI server
# ----------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
