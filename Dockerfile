FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]