# tokentoken/utils.py
import tiktoken
import warnings
from typing import List, Dict, Any, Optional
import math

def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """Accurately count tokens using tiktoken (cl100k_base)."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def check_breakeven_threshold(token_count: int):
    """Warning based on Section 4.3 of the paper."""
    if token_count < 300:
        warnings.warn(
            "Chain-of-Thought Tax Warning: Input is very short. "
            "The downstream LLM may use more output tokens to decode the compressed text "
            "than you save on input tokens. Best used for > 1,000 tokens."
        )

def chunk_text(text: str, chunk_size_tokens: int = 150000, overlap_tokens: int = 1000) -> List[str]:
    """Based on Appendix E.4.2 (Extending Context Window).
    
    Splits text into chunks that fit within the model's context window.
    Uses token-based chunking with optional overlap for continuity.
    Overlap is clamped to half the chunk size so iteration always advances.
    """
    if not text.strip():
        return []
    
    chunk_size_tokens = max(1, chunk_size_tokens)
    overlap_tokens = max(0, overlap_tokens)
    
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= chunk_size_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        
        # Try to break at sentence or paragraph boundaries for better context
        if end < len(tokens):
            # Look back for a natural break point
            chunk_tokens = tokens[start:end]
            chunk_text_decoded = encoding.decode(chunk_tokens)
            
            # Try to find a paragraph break
            last_paragraph_break = chunk_text_decoded.rfind("\n\n")
            if last_paragraph_break > chunk_size_tokens * 0.8:  # At least 80% of chunk
                chunk_text_decoded = chunk_text_decoded[:last_paragraph_break]
                end = start + len(encoding.encode(chunk_text_decoded))
            else:
                # Try to find a sentence break
                last_sentence_break = chunk_text_decoded.rfind(". ")
                if last_sentence_break > chunk_size_tokens * 0.8:
                    chunk_text_decoded = chunk_text_decoded[:last_sentence_break + 1]
                    end = start + len(encoding.encode(chunk_text_decoded))
        
        chunks.append(encoding.decode(tokens[start:end]))
        
        if end >= len(tokens):
            break
        
        # Move to next chunk with overlap; clamp so start always advances
        step = end - start
        start = end - min(overlap_tokens, step // 2)
    
    return chunks

def calculate_compression_metrics(original_tokens: int, compressed_tokens: int) -> Dict[str, float]:
    """Calculate compression metrics based on the paper's evaluation criteria."""
    savings_pct = round((1 - (compressed_tokens / original_tokens)) * 100, 2) if original_tokens > 0 else 0
    retention_ratio = round(compressed_tokens / original_tokens, 4) if original_tokens > 0 else 0
    
    # Section 4.3: Chain-of-Thought Tax estimation
    # Based on paper's finding that stronger compression often increases chain-of-thought tokens
    cot_tax_estimate = max(0, (1 - retention_ratio) * 0.5)  # Rough estimate
    
    return {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "savings_pct": savings_pct,
        "retention_ratio": retention_ratio,
        "cot_tax_estimate": cot_tax_estimate
    }

def estimate_readability(text: str) -> Dict[str, Any]:
    """Estimate readability metrics based on Section E.1 of the paper.
    
    Returns Dale-Chall-like score approximation and difficult word ratio.
    """
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return {
            "dale_chall_estimate": 0,
            "difficult_word_ratio": 0,
            "word_count": 0
        }
    
    # Simple approximation of difficult words (words > 6 characters)
    difficult_words = [w for w in words if len(w) > 6]
    difficult_word_ratio = len(difficult_words) / total_words
    
    # Approximate Dale-Chall score based on word length distribution
    # Higher score = lower readability
    avg_word_length = sum(len(w) for w in words) / total_words
    dale_chall_estimate = 10 + (avg_word_length - 4) * 2 + difficult_word_ratio * 10
    
    return {
        "dale_chall_estimate": round(dale_chall_estimate, 2),
        "difficult_word_ratio": round(difficult_word_ratio, 4),
        "word_count": total_words
    }

def calculate_token_efficiency(text: str, model_name: str = "gpt-4o") -> Dict[str, float]:
    """Calculate token efficiency metrics for compression analysis."""
    tokens = count_tokens(text, model_name)
    words = text.split()
    word_count = len(words)
    
    if word_count == 0:
        return {
            "tokens_per_word": 0,
            "chars_per_token": 0,
            "compression_potential": 0
        }
    
    tokens_per_word = tokens / word_count
    chars_per_token = len(text) / tokens if tokens > 0 else 0
    
    # Estimate potential for further compression based on information density
    # Based on paper's finding that compression achieves ~27.9% retention
    compression_potential = max(0, 1 - (tokens_per_word / 2))  # Rough estimate
    
    return {
        "tokens_per_word": round(tokens_per_word, 2),
        "chars_per_token": round(chars_per_token, 2),
        "compression_potential": round(compression_potential, 4)
    }

def validate_compression_result(original: str, compressed: str, provider: str, model: str) -> Dict[str, Any]:
    """Validate compression result against paper's evaluation criteria."""
    original_tokens = count_tokens(original)
    compressed_tokens = count_tokens(compressed)
    
    metrics = calculate_compression_metrics(original_tokens, compressed_tokens)
    readability = estimate_readability(compressed)
    efficiency = calculate_token_efficiency(compressed)
    
    # Paper Section 4.3: Chain-of-Thought Tax warning
    cot_warning = None
    if metrics["retention_ratio"] < 0.2:  # Less than 20% retention
        cot_warning = "High compression may trigger Chain-of-Thought Tax (Section 4.3)"
    
    return {
        "compression_metrics": metrics,
        "readability_metrics": readability,
        "efficiency_metrics": efficiency,
        "provider": provider,
        "model": model,
        "cot_warning": cot_warning
    }
