FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy utility modules first
COPY utils/ ./utils/

# Copy new package directories
COPY agents/ ./agents/
COPY models/ ./models/
COPY services/ ./services/

# Copy core application files (entry points)
COPY unified_app.py .
COPY app.py .
COPY stock_app.py .

# Create cache directory
RUN mkdir -p .cache

# Expose Gradio port
EXPOSE 7860

# Health check - marks container unhealthy after 3 failed checks
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=5)" || exit 1

# Run the Research Agent Hub
CMD ["python", "unified_app.py"]
