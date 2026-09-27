# tests/conftest.py
import pytest
import tempfile
import os


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run each test inside a temp cwd.

    CLI commands default to outputs like dense.txt / compressed_long.txt in the
    current directory; without this fixture a test run overwrites real artifacts
    in the repo root.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """Large language models (LLMs) have become a dominant interface in contemporary intelligent systems. 
Since GPT-3 demonstrated strong few-shot generalization through text-based prompting, 
the field has largely followed a unified paradigm: knowledge is represented in natural language, 
instructions are issued in natural language, and model outputs are returned in natural language."""


@pytest.fixture
def compressed_text():
    """Sample compressed text for testing."""
    return """LLM>dominant iface. GPT-3 fewshot. Paradigm: knowledge∈NL, instr∈NL, output∈NL."""


@pytest.fixture
def tmp_text_file(tmp_path):
    """Create a temporary text file for CLI testing."""
    def _create(content, filename="test.txt"):
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    return _create
