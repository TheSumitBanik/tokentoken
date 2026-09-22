# tests/test_prompts.py
import pytest
from tokentoken.prompts import CompressionMode, PROMPTS


class TestCompressionMode:
    def test_all_modes_exist(self):
        # Check that all 13 BT-P modes exist
        assert CompressionMode.DEFAULT.value == "default"
        assert CompressionMode.BT_P1.value == "bt_p1"
        assert CompressionMode.BT_P13.value == "bt_p13"
    
    def test_all_modes_have_prompts(self):
        for mode in CompressionMode:
            assert mode in PROMPTS, f"Missing prompt for {mode.value}"
    
    def test_prompt_content_quality(self):
        # All prompts should mention compression
        for mode, prompt in PROMPTS.items():
            assert len(prompt) > 100, f"Prompt for {mode.value} is too short"
            # Should contain compression-related keywords
            assert any(keyword in prompt.lower() for keyword in [
                "compress", "babel", "llm", "token"
            ]), f"Prompt for {mode.value} missing compression keywords"


class TestPrompts:
    def test_default_prompt(self):
        prompt = PROMPTS[CompressionMode.DEFAULT]
        assert "compress" in prompt.lower()
        assert "omnilingual" in prompt.lower()
        assert "symbolic" in prompt.lower()
    
    def test_prompt_variants_exist(self):
        # All 13 variants should be present
        variants = [
            CompressionMode.BT_P1, CompressionMode.BT_P2, CompressionMode.BT_P3,
            CompressionMode.BT_P4, CompressionMode.BT_P5, CompressionMode.BT_P6,
            CompressionMode.BT_P7, CompressionMode.BT_P8, CompressionMode.BT_P9,
            CompressionMode.BT_P10, CompressionMode.BT_P11, CompressionMode.BT_P12,
            CompressionMode.BT_P13
        ]
        for variant in variants:
            assert variant in PROMPTS
    
    def test_prompt_lengths(self):
        # All prompts should be substantial
        for mode, prompt in PROMPTS.items():
            word_count = len(prompt.split())
            assert word_count >= 20, f"Prompt for {mode.value} too short: {word_count} words"
