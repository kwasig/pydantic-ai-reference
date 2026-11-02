"""Reference PydanticAI agent running on Nebius Token Factory."""

from .agent import ModelRecommendation, build_agent
from .config import Settings, get_settings
from .provider import make_model

__all__ = ["ModelRecommendation", "Settings", "build_agent", "get_settings", "make_model"]
__version__ = "0.1.0"
