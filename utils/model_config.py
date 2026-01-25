"""
Model configuration for Research Agent Hub.
Defines available models and their pricing for cost estimation.
"""

from typing import Dict, Tuple

# Available models with display names
AVAILABLE_MODELS = {
    "claude-sonnet-4-20250514": "Claude Sonnet 4 (Fast, Cost-Effective)",
    "claude-opus-4-20250514": "Claude Opus 4 (Most Capable)",
}

# Model pricing per 1M tokens (input, output)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),      # $3/1M input, $15/1M output
    "claude-opus-4-20250514": (15.0, 75.0),       # $15/1M input, $75/1M output
}

# Default model
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def get_model_choices() -> list:
    """Return list of (display_name, model_id) tuples for dropdown."""
    return [(name, model_id) for model_id, name in AVAILABLE_MODELS.items()]


def get_model_display_name(model_id: str) -> str:
    """Get display name for a model ID."""
    return AVAILABLE_MODELS.get(model_id, model_id)


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost based on model and token counts."""
    pricing = MODEL_PRICING.get(model_id, MODEL_PRICING[DEFAULT_MODEL])
    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost
