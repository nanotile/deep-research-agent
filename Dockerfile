FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY unified_app.py .
COPY deep_research_agent.py .
COPY stock_research_agent.py .
COPY stock_data_fetchers.py .
COPY stock_data_models.py .
COPY market_context_2026.py .

# Expose Gradio port
EXPOSE 7860

# Run the Research Agent Hub
CMD ["python", "unified_app.py"]
