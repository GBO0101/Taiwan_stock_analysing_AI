"""Unit tests for LLMClient."""

import json
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field

from classifier.llm_client import (
    LLMClient,
    LLMError,
    LLMNetworkError,
    LLMTimeoutError,
    LLMAuthError,
    LLMOutputError,
)
from classifier.models import BoundaryResult, ClassificationType


class TestResponseModel(BaseModel):
    """Test response model."""

    value: str = Field(..., description="Test value")
    confidence: float = Field(..., ge=0.0, le=1.0)


class TestLLMClient:
    """Test LLMClient functionality."""

    def test_init_with_defaults(self, monkeypatch):
        """Test initialization with default settings (env-independent)."""
        import classifier.config as config_mod

        monkeypatch.setattr(config_mod.settings, "llm_api_key", "test_key_for_testing")
        monkeypatch.setattr(config_mod.settings, "llm_model", "gpt-4o-mini")
        monkeypatch.setattr(config_mod.settings, "llm_timeout", 30)
        monkeypatch.setattr(
            config_mod.settings, "llm_base_url", "https://api.openai.com/v1"
        )
        client = LLMClient()
        assert client.api_key == "test_key_for_testing"
        assert client.model == "gpt-4o-mini"
        assert client.timeout == 30
        assert client.base_url == "https://api.openai.com/v1"

    def test_init_with_overrides(self):
        """Test initialization with custom parameters."""
        client = LLMClient(
            api_key="custom_key",
            model="gpt-4",
            timeout=60,
            base_url="http://localhost:11434/v1",
        )
        assert client.api_key == "custom_key"
        assert client.model == "gpt-4"
        assert client.timeout == 60
        assert client.base_url == "http://localhost:11434/v1"

    @patch("classifier.llm_client.OpenAI")
    def test_extract_structured_success(self, mock_openai_class):
        """Test successful structured extraction via JSON mode."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [
            Mock(message=Mock(content='{"value": "test", "confidence": 0.9}'))
        ]
        mock_client.chat.completions.create.return_value = mock_completion

        client = LLMClient()
        result = client.extract_structured("test prompt", TestResponseModel)

        assert isinstance(result, TestResponseModel)
        assert result.value == "test"
        assert result.confidence == 0.9
        mock_client.chat.completions.create.assert_called_once()
        # JSON mode must be requested so non-OpenAI endpoints work too
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["response_format"] == {"type": "json_object"}

    @patch("classifier.llm_client.OpenAI")
    def test_extract_structured_empty_content(self, mock_openai_class):
        """Test handling of empty model output."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content=None))]
        mock_client.chat.completions.create.return_value = mock_completion

        client = LLMClient()
        with pytest.raises(LLMOutputError, match="empty content"):
            client.extract_structured("test prompt", TestResponseModel)

    @patch("classifier.llm_client.OpenAI")
    def test_extract_structured_invalid_json(self, mock_openai_class):
        """Test handling of non-JSON model output."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="not json at all"))]
        mock_client.chat.completions.create.return_value = mock_completion

        client = LLMClient()
        with pytest.raises(LLMOutputError, match="Failed to parse JSON"):
            client.extract_structured("test prompt", TestResponseModel)

    @patch("classifier.llm_client.OpenAI")
    def test_extract_structured_validation_error(self, mock_openai_class):
        """Test handling of JSON that fails Pydantic validation."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_completion = Mock()
        # confidence out of range -> ValidationError from model_validate
        mock_completion.choices = [
            Mock(message=Mock(content='{"value": "test", "confidence": 5.0}'))
        ]
        mock_client.chat.completions.create.return_value = mock_completion

        client = LLMClient()
        with pytest.raises(LLMOutputError, match="Output validation failed"):
            client.extract_structured("test prompt", TestResponseModel)

    @patch("classifier.llm_client.OpenAI")
    def test_extract_structured_timeout(self, mock_openai_class):
        """Test timeout error handling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = TimeoutError(
            "Request timed out"
        )

        client = LLMClient()
        with pytest.raises(LLMTimeoutError, match="Request timeout"):
            client.extract_structured("test prompt", TestResponseModel)

    @patch("classifier.llm_client.OpenAI")
    def test_extract_structured_auth_error(self, mock_openai_class):
        """Test authentication error handling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception(
            "401 Unauthorized"
        )

        client = LLMClient()
        with pytest.raises(LLMAuthError, match="Authentication failed"):
            client.extract_structured("test prompt", TestResponseModel)

    @patch("classifier.llm_client.OpenAI")
    def test_extract_structured_network_error(self, mock_openai_class):
        """Test network error handling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception(
            "Connection refused"
        )

        client = LLMClient()
        with pytest.raises(LLMNetworkError, match="Network error"):
            client.extract_structured("test prompt", TestResponseModel)

    @patch("classifier.llm_client.OpenAI")
    def test_extract_structured_generic_error(self, mock_openai_class):
        """Test generic error handling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Unknown error")

        client = LLMClient()
        with pytest.raises(LLMOutputError, match="LLM error"):
            client.extract_structured("test prompt", TestResponseModel)

    def test_exception_hierarchy(self):
        """Test exception class hierarchy."""
        assert issubclass(LLMNetworkError, LLMError)
        assert issubclass(LLMTimeoutError, LLMError)
        assert issubclass(LLMAuthError, LLMError)
        assert issubclass(LLMOutputError, LLMError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
