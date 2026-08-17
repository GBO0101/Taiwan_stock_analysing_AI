"""Vendor-neutral LLM client with structured output extraction.

Uses the OpenAI-compatible Chat Completions API (JSON mode) so any provider
exposing ``/v1/chat/completions`` (OpenAI, Ollama, vLLM, Groq, DeepSeek,
OpenRouter, LM Studio, ...) works by pointing ``LLM_BASE_URL`` at it.
"""

import json
from typing import Type, TypeVar

from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from classifier.config import settings

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class LLMNetworkError(LLMError):
    """Network-related errors."""

    pass


class LLMTimeoutError(LLMError):
    """Request timeout errors."""

    pass


class LLMAuthError(LLMError):
    """Authentication errors."""

    pass


class LLMOutputError(LLMError):
    """Malformed or invalid output errors."""

    pass


class LLMClient:
    """OpenAI-compatible LLM client for structured output extraction."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        base_url: str | None = None,
    ):
        """Initialize LLM client.

        Args:
            api_key: LLM API key (defaults to settings)
            model: Model name (defaults to settings)
            timeout: Request timeout in seconds (defaults to settings)
            base_url: OpenAI-compatible endpoint base URL (defaults to settings)
        """
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout
        self.base_url = base_url or settings.llm_base_url

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @staticmethod
    def _schema_hint(response_model: Type[BaseModel]) -> str:
        """Build a compact JSON-schema hint for the response model."""
        schema = response_model.model_json_schema()
        hint = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        return json.dumps(hint, ensure_ascii=False)

    def extract_structured(self, prompt: str, response_model: Type[T]) -> T:
        """Extract structured output from an OpenAI-compatible LLM.

        Uses JSON mode (``response_format={"type": "json_object"}``) plus a
        schema hint, then validates the returned JSON with the Pydantic model.
        This avoids the OpenAI-only ``beta.chat.completions.parse`` so any
        OpenAI-compatible provider works.

        Args:
            prompt: The prompt to send to the LLM
            response_model: Pydantic model class for structured output

        Returns:
            Validated instance of response_model

        Raises:
            LLMNetworkError: Network failure
            LLMTimeoutError: Request timeout
            LLMAuthError: Authentication failure
            LLMOutputError: Malformed output or schema validation failure
        """
        system_prompt = (
            "You are a precise data extraction assistant. "
            "Respond with ONLY a single valid JSON object that matches the "
            "following JSON schema. Do not include any explanatory text, "
            "markdown code fences, or content outside the JSON object.\n\n"
            f"Schema:\n{self._schema_hint(response_model)}"
        )
        try:
            completion: ChatCompletion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = completion.choices[0].message.content
            if not content or not content.strip():
                raise LLMOutputError("LLM returned empty content")
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMOutputError(f"Failed to parse JSON output: {e}") from e
            try:
                return response_model.model_validate(data)
            except ValidationError as e:
                raise LLMOutputError(f"Output validation failed: {e}") from e
        except LLMOutputError:
            raise
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                raise LLMTimeoutError(f"Request timeout: {error_msg}") from e
            elif (
                "authentication" in error_msg.lower()
                or "unauthorized" in error_msg.lower()
                or "401" in error_msg
            ):
                raise LLMAuthError(f"Authentication failed: {error_msg}") from e
            elif (
                "connection" in error_msg.lower()
                or "network" in error_msg.lower()
                or "dns" in error_msg.lower()
            ):
                raise LLMNetworkError(f"Network error: {error_msg}") from e
            else:
                raise LLMOutputError(f"LLM error ({error_type}): {error_msg}") from e
