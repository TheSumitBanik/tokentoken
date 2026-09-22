# tests/test_cli.py
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from tokentoken.cli import app


runner = CliRunner()


@pytest.fixture
def mock_compress():
    with patch('tokentoken.cli.TokenToken') as mock:
        instance = mock.return_value
        instance.compress.return_value = MagicMock(
            dense_text="compressed text",
            original_tokens=1000,
            compressed_tokens=300,
            savings_pct=70.0,
            retention_ratio=0.3,
            readability_metrics={"dale_chall_estimate": 15.0, "difficult_word_ratio": 0.4},
            efficiency_metrics={"tokens_per_word": 1.5, "chars_per_token": 4.2},
            validation={"cot_warning": None}
        )
        yield mock


class TestCLIHelp:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Compile human prompts into LLM machine language" in result.output
    
    def test_compress_help(self):
        result = runner.invoke(app, ["compress", "--help"])
        assert result.exit_code == 0
        assert "Compress text using TokenToken" in result.output
    
    def test_decompress_help(self):
        result = runner.invoke(app, ["decompress", "--help"])
        assert result.exit_code == 0
    
    def test_list_modes_help(self):
        result = runner.invoke(app, ["list-modes", "--help"])
        assert result.exit_code == 0


class TestListModes:
    def test_list_modes(self):
        result = runner.invoke(app, ["list-modes"])
        assert result.exit_code == 0
        assert "TokenToken Compression Modes" in result.output
        assert "bt_p1" in result.output
        assert "bt_p13" in result.output


class TestEstimate:
    def test_estimate(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test file with some content.")
        
        result = runner.invoke(app, ["estimate", str(test_file)])
        assert result.exit_code == 0
        assert "tokens" in result.output.lower()


class TestCompress:
    def test_compress_basic(self, tmp_path, mock_compress):
        test_file = tmp_path / "input.txt"
        test_file.write_text("This is a test sentence that needs compression.")
        
        result = runner.invoke(app, [
            "compress",
            str(test_file),
            "--provider", "openai",
            "--model", "gpt-4"
        ])
        assert result.exit_code == 0
        assert "TokenToken Compression Results" in result.output
    
    def test_compress_with_mode(self, tmp_path, mock_compress):
        test_file = tmp_path / "input.txt"
        test_file.write_text("Test content for compression.")
        
        result = runner.invoke(app, [
            "compress",
            str(test_file),
            "--mode", "bt_p5"
        ])
        assert result.exit_code == 0
    
    def test_compress_invalid_mode(self, tmp_path):
        test_file = tmp_path / "input.txt"
        test_file.write_text("Test content.")
        
        result = runner.invoke(app, [
            "compress",
            str(test_file),
            "--mode", "invalid_mode"
        ])
        assert result.exit_code == 0  # Should show error message
        assert "Invalid mode" in result.output

    def test_compress_inline_text(self, mock_compress):
        result = runner.invoke(app, [
            "compress",
            "--text", "This is verbose text passed inline for compression.",
            "--provider", "openai",
            "--model", "gpt-4"
        ])
        assert result.exit_code == 0
        assert "TokenToken Compression Results" in result.output

    def test_compress_inline_text_short_flag(self, mock_compress):
        result = runner.invoke(app, ["compress", "-t", "Short flag inline text."])
        assert result.exit_code == 0
        assert "TokenToken Compression Results" in result.output

    def test_compress_file_and_text_rejected(self, tmp_path):
        test_file = tmp_path / "input.txt"
        test_file.write_text("Some content.")
        
        result = runner.invoke(app, ["compress", str(test_file), "--text", "inline text"])
        assert result.exit_code == 1
        assert "not both" in result.output

    def test_compress_no_input(self):
        result = runner.invoke(app, ["compress"])
        assert result.exit_code == 1
        assert "No input" in result.output

    def test_base_url_option_passed_to_sdk(self, mock_compress):
        result = runner.invoke(app, [
            "compress",
            "--text", "hello world",
            "--provider", "openai",
            "--model", "my-model",
            "--base-url", "http://localhost:1234/v1"
        ])
        assert result.exit_code == 0
        assert mock_compress.call_args.kwargs["base_url"] == "http://localhost:1234/v1"


class TestDecompress:
    def test_decompress(self, tmp_path):
        test_file = tmp_path / "compressed.txt"
        test_file.write_text("Compressed text here.")
        
        with patch('tokentoken.cli.TokenToken') as mock:
            instance = mock.return_value
            instance.decompress.return_value = "This is the decompressed text."
            
            result = runner.invoke(app, [
                "decompress",
                str(test_file),
                "--provider", "openai",
                "--model", "gpt-4"
            ])
            assert result.exit_code == 0
            assert "Decompression Result" in result.output


class TestAnalyze:
    def test_analyze_basic(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test file for analysis.")
        
        result = runner.invoke(app, ["analyze", str(test_file)])
        assert result.exit_code == 0
        assert "Text Analysis Results" in result.output
    
    def test_analyze_with_compressed(self, tmp_path):
        test_file = tmp_path / "original.txt"
        test_file.write_text("This is original text for testing compression analysis.")
        
        compressed_file = tmp_path / "compressed.txt"
        compressed_file.write_text("Original text testing.")
        
        result = runner.invoke(app, [
            "analyze",
            str(test_file),
            "--compressed", str(compressed_file)
        ])
        assert result.exit_code == 0
        assert "Token Reduction" in result.output


class TestAgentMemory:
    def test_agent_memory(self, tmp_path):
        # Create memory files
        memory_dir = tmp_path / "memories"
        memory_dir.mkdir()
        (memory_dir / "memory1.txt").write_text("Session 1: User asked about weather")
        (memory_dir / "memory2.txt").write_text("Session 2: User asked about restaurants")
        
        with patch('tokentoken.cli.TokenToken') as mock:
            instance = mock.return_value
            instance.compress_for_agent_memory.return_value = MagicMock(
                compressed_memories=["compressed1", "compressed2"],
                original_token_count=100,
                compressed_token_count=30,
                memory_count=2,
                avg_compression_ratio=0.3
            )
            
            result = runner.invoke(app, [
                "agent-memory",
                str(memory_dir),
                "--provider", "openai",
                "--model", "gpt-4"
            ])
            assert result.exit_code == 0
            assert "Agent Memory Compression Results" in result.output


class TestMultiAgent:
    def test_multi_agent(self, tmp_path):
        message_file = tmp_path / "message.txt"
        message_file.write_text("Please analyze the user data and provide insights.")
        
        with patch('tokentoken.cli.TokenToken') as mock:
            instance = mock.return_value
            instance.compress_for_multi_agent.return_value = MagicMock(
                sender="agent_1",
                receiver="agent_2",
                content="compressed message",
                compressed=True,
                original_tokens=50,
                compressed_tokens=15
            )
            
            result = runner.invoke(app, [
                "multi-agent",
                str(message_file),
                "--sender", "agent_1",
                "--receiver", "agent_2",
                "--provider", "openai",
                "--model", "gpt-4"
            ])
            assert result.exit_code == 0
            assert "Multi-Agent Message Compression Results" in result.output


class TestCrossModel:
    def test_cross_model(self, tmp_path):
        compressed_file = tmp_path / "compressed.txt"
        compressed_file.write_text("Compressed text.")
        
        with patch('tokentoken.cli.TokenToken') as mock:
            instance = mock.return_value
            instance.check_cross_model_compatibility.return_value = MagicMock(
                compressor_model="gpt-4",
                compressor_provider="openai",
                reader_model="gemini-pro",
                reader_provider="gemini",
                compatibility_score=0.85,
                retention_ratio=1.0,
                notes="Tested via simple comprehension check"
            )
            
            result = runner.invoke(app, [
                "cross-model",
                str(compressed_file),
                "--reader-provider", "gemini",
                "--reader-model", "gemini-pro",
                "--provider", "openai",
                "--model", "gpt-4"
            ])
            assert result.exit_code == 0
            assert "Cross-Model Compatibility Results" in result.output


class TestExtendContext:
    def test_extend_context(self, tmp_path):
        long_file = tmp_path / "long_document.txt"
        long_file.write_text("This is a very long document. " * 1000)
        
        with patch('tokentoken.cli.TokenToken') as mock:
            instance = mock.return_value
            instance.extend_context_window.return_value = "Compressed long document."
            
            result = runner.invoke(app, [
                "extend-context",
                str(long_file),
                "--target-tokens", "1000",
                "--provider", "openai",
                "--model", "gpt-4"
            ])
            assert result.exit_code == 0
            assert "Context Window Extension Results" in result.output


class TestErrorHandling:
    def test_compress_missing_file_is_clean(self):
        result = runner.invoke(app, ["compress", "nonexistent_file.txt"])
        assert result.exit_code == 1
        assert "file not found" in result.output.lower()
        assert "Traceback" not in result.output
        assert not isinstance(result.exception, (FileNotFoundError, OSError))

    def test_decompress_missing_file_is_clean(self):
        result = runner.invoke(app, ["decompress", "nonexistent_file.txt"])
        assert result.exit_code == 1
        assert "file not found" in result.output.lower()
        assert "Traceback" not in result.output
        assert not isinstance(result.exception, (FileNotFoundError, OSError))

    def test_estimate_missing_file_is_clean(self):
        result = runner.invoke(app, ["estimate", "nonexistent_file.txt"])
        assert result.exit_code == 1
        assert "file not found" in result.output.lower()
        assert "Traceback" not in result.output
        assert not isinstance(result.exception, (FileNotFoundError, OSError))

    def test_api_error_is_clean(self):
        with patch('tokentoken.cli.TokenToken') as mock:
            instance = mock.return_value
            instance.compress.side_effect = RuntimeError("provider unreachable")
            
            result = runner.invoke(app, ["compress", "--text", "hello world"])
            assert result.exit_code == 1
            assert "provider unreachable" in result.output
            assert "Traceback" not in result.output
            assert not isinstance(result.exception, RuntimeError)
