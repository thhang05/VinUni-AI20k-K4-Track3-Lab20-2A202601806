"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# Rough public pricing (USD per 1K tokens) used only for benchmark cost estimates.
_PRICING_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4.1-mini": (0.0004, 0.0016),
}
_DEFAULT_PRICING = (0.0005, 0.0015)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    input_price, output_price = _PRICING_PER_1K_TOKENS.get(model, _DEFAULT_PRICING)
    return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price


class LLMClient:
    """OpenAI-backed LLM client with retry/timeout handling."""

    def __init__(self) -> None:
        self._settings = get_settings()
        if not self._settings.openai_api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not configured. Set it in .env before using LLMClient."
            )
        self._client = OpenAI(
            api_key=self._settings.openai_api_key, timeout=self._settings.timeout_seconds
        )
        self._model = self._settings.openai_model

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIError)),
    )
    def _call(self, system_prompt: str, user_prompt: str, temperature: float) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost = _estimate_cost(self._model, input_tokens, output_tokens)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def complete(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> LLMResponse:
        """Return a model completion, retrying transient failures with backoff."""

        try:
            return self._call(system_prompt, user_prompt, temperature)
        except (APITimeoutError, RateLimitError, APIError) as exc:
            logger.error("LLM call failed after retries: %s", exc)
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc
