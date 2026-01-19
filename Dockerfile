FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY deep_research_agent.py .

# Expose Gradio port
EXPOSE 7860

# Run the web interface
CMD ["python", "app.py"]
