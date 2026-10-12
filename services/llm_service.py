import os
import json
from typing import Optional, Dict, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMService:
    """
    Configurable LLM provider service.
    Reads API keys strictly from environment variables.
    Provides structured JSON parsing and graceful failure handling.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._client = None

        if self.is_configured() and OpenAI:
            try:
                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    def is_configured(self) -> bool:
        """
        Returns True if a non-placeholder API key is configured.
        """
        if not self.api_key:
            return False
        # Filter out obvious template placeholders
        placeholder_indicators = ["your_openai_api_key_here", "placeholder", "xxx"]
        return not any(p in self.api_key.lower() for p in placeholder_indicators)

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2
    ) -> str:
        """
        Calls the LLM provider for text completion.
        Raises RuntimeError if LLM is unconfigured or request fails.
        """
        if not self.is_configured():
            raise RuntimeError(
                "OpenAI API key is not configured in environment (.env). "
                "Please configure OPENAI_API_KEY to enable LLM generation."
            )

        if not self._client:
            raise RuntimeError("OpenAI client could not be initialized.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prompts the LLM and parses the response as structured JSON.
        """
        raw_output = self.complete(prompt, system_prompt=system_prompt, json_mode=True)
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError as err:
            raise RuntimeError(f"Failed to parse LLM response as JSON: {raw_output[:200]}") from err
