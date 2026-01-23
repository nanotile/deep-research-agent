FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy utility modules first
COPY utils/ ./utils/

# Copy core application files
COPY unified_app.py .
COPY deep_research_agent.py .
COPY stock_research_agent.py .
COPY stock_data_fetchers.py .
COPY stock_data_models.py .
COPY market_context_2026.py .

# Copy Phase 3 agents
COPY sector_research_agent.py .
COPY competitor_agent.py .
COPY portfolio_agent.py .

# Copy Phase 4 agents
COPY earnings_agent.py .
COPY alert_system.py .

# Create cache directory
RUN mkdir -p .cache

# Expose Gradio port
EXPOSE 7860

# Run the Research Agent Hub
CMD ["python", "unified_app.py"]
