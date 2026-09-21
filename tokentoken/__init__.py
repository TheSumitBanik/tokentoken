"""tokentoken - Compile human prompts into LLM machine language."""

from .sdk import TokenToken, CompressionResult, CrossModelCompatibility, AgentMemoryResult, MultiAgentMessage
from .prompts import CompressionMode
from .utils import (
    count_tokens, 
    check_breakeven_threshold, 
    chunk_text, 
    calculate_compression_metrics,
    estimate_readability,
    calculate_token_efficiency,
    validate_compression_result
)

__version__ = "1.0.0"
__author__ = "Sumit Banik"

__all__ = [
    "TokenToken",
    "CompressionResult",
    "CrossModelCompatibility", 
    "AgentMemoryResult",
    "MultiAgentMessage",
    "CompressionMode",
    "count_tokens",
    "check_breakeven_threshold",
    "chunk_text",
    "calculate_compression_metrics",
    "estimate_readability",
    "calculate_token_efficiency",
    "validate_compression_result"
]