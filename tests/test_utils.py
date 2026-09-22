# tests/test_utils.py
import pytest
from tokentoken.utils import (
    count_tokens,
    check_breakeven_threshold,
    chunk_text,
    calculate_compression_metrics,
    estimate_readability,
    calculate_token_efficiency,
    validate_compression_result
)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0
    
    def test_simple_text(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Simple text should be small
    
    def test_longer_text(self):
        text = "This is a longer text that should have more tokens. " * 10
        tokens = count_tokens(text)
        assert tokens > 20
    
    def test_unicode_text(self):
        tokens = count_tokens("你好世界")
        assert tokens > 0


class TestCheckBreakevenThreshold:
    def test_short_text_warning(self):
        with pytest.warns(UserWarning, match="Chain-of-Thought Tax"):
            check_breakeven_threshold(100)
    
    def test_medium_text_no_warning(self):
        # Should not warn for text > 300 tokens
        check_breakeven_threshold(500)
    
    def test_long_text_no_warning(self):
        check_breakeven_threshold(10000)


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []
    
    def test_short_text_no_chunking(self):
        text = "Short text"
        chunks = chunk_text(text, chunk_size_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    @pytest.mark.slow
    def test_long_text_chunking(self):
        # Create text longer than chunk size
        # Use shorter text to avoid timeout
        text = "This is a sentence. " * 100
        chunks = chunk_text(text, chunk_size_tokens=50)
        assert len(chunks) > 1
    
    def test_chunks_are_strings(self):
        text = "Test text"
        chunks = chunk_text(text, chunk_size_tokens=1000)
        assert all(isinstance(chunk, str) for chunk in chunks)


class TestCalculateCompressionMetrics:
    def test_basic_metrics(self):
        metrics = calculate_compression_metrics(1000, 300)
        assert metrics["original_tokens"] == 1000
        assert metrics["compressed_tokens"] == 300
        assert metrics["savings_pct"] == 70.0
        assert metrics["retention_ratio"] == 0.3
    
    def test_no_compression(self):
        metrics = calculate_compression_metrics(1000, 1000)
        assert metrics["savings_pct"] == 0.0
        assert metrics["retention_ratio"] == 1.0
    
    def test_zero_original_tokens(self):
        metrics = calculate_compression_metrics(0, 0)
        assert metrics["savings_pct"] == 0
        assert metrics["retention_ratio"] == 0


class TestEstimateReadability:
    def test_empty_text(self):
        result = estimate_readability("")
        assert result["dale_chall_estimate"] == 0
        assert result["difficult_word_ratio"] == 0
        assert result["word_count"] == 0
    
    def test_simple_text(self):
        text = "The cat sat on the mat."
        result = estimate_readability(text)
        assert result["word_count"] == 6
        assert result["difficult_word_ratio"] >= 0
        assert result["difficult_word_ratio"] <= 1
    
    def test_complex_text(self):
        # Text with many long words
        text = "The anthropomorphized concatenation of unprecedented configurations demonstrates sophistication."
        result = estimate_readability(text)
        assert result["difficult_word_ratio"] > 0.5


class TestCalculateTokenEfficiency:
    def test_empty_text(self):
        result = calculate_token_efficiency("")
        assert result["tokens_per_word"] == 0
        assert result["chars_per_token"] == 0
    
    def test_simple_text(self):
        text = "Hello world"
        result = calculate_token_efficiency(text)
        assert result["tokens_per_word"] > 0
        assert result["chars_per_token"] > 0
    
    def test_efficiency_metrics(self):
        text = "This is a test sentence with multiple words."
        result = calculate_token_efficiency(text)
        assert result["tokens_per_word"] > 1  # Usually > 1 token per word
        assert result["chars_per_token"] > 0


class TestValidateCompressionResult:
    def test_validation_structure(self):
        original = "Original text here"
        compressed = "Original text"
        result = validate_compression_result(original, compressed, "openai", "gpt-4")
        
        assert "compression_metrics" in result
        assert "readability_metrics" in result
        assert "efficiency_metrics" in result
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4"
    
    def test_compression_warning(self):
        # Very high compression should trigger warning
        original = "A" * 1000
        compressed = "A"
        result = validate_compression_result(original, compressed, "openai", "gpt-4")
        assert result["cot_warning"] is not None
        assert "Chain-of-Thought Tax" in result["cot_warning"]
