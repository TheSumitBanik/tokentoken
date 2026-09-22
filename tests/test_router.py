# tests/test_router.py
import pytest
from unittest.mock import patch, MagicMock
from tokentoken.router import execute_compression


def _mock_openai_response():
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content="dense output"))]
    )


class TestOpenAICompatibleEndpoint:
    def test_base_url_forwarded_to_client(self):
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = _mock_openai_response()
            result = execute_compression(
                provider="openai",
                model="my-model",
                system_prompt="system",
                user_text="user text",
                api_key="key",
                base_url="http://localhost:1234/v1",
            )
        assert result == "dense output"
        mock_client.assert_called_once_with(
            api_key="key", base_url="http://localhost:1234/v1"
        )

    def test_base_url_defaults_to_openai_endpoint(self):
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = _mock_openai_response()
            execute_compression(
                provider="openai",
                model="gpt-4",
                system_prompt="s",
                user_text="u",
                api_key="k",
            )
        mock_client.assert_called_once_with(api_key="k", base_url=None)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            execute_compression(
                provider="nope", model="m", system_prompt="s", user_text="u"
            )
