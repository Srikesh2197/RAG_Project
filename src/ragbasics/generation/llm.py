"""Generators: turn a system prompt and a user prompt into an answer."""

from dataclasses import dataclass
from typing import Any, Protocol

from ragbasics.registry import register


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    cached: bool = False  # read from the disk cache, so nothing was billed


class Generator(Protocol):
    model: str

    def generate(self, system: str, user: str) -> Generation: ...


@register("generator", "anthropic")
class AnthropicGenerator:
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        max_tokens: int = 4096,
        effort: str = "low",
        client: Any = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = client
        # The settings that change the output; part of the disk-cache key.
        self.cache_params = {"max_tokens": max_tokens, "effort": effort}

    @property
    def client(self) -> Any:
        # Created on first use, so building a pipeline does not need an API key.
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, system: str, user: str) -> Generation:
        # This model does not accept a temperature. `effort` sets how much it reasons
        # before answering; reasoning is billed as output tokens.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return Generation(
            text=text.strip(),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason or "",
        )


@register("generator", "openai")
class OpenAIGenerator:
    """An OpenAI chat model. Used as the judge (Stage 2), from a different vendor than
    the generator so that no model grades its own family."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 2000,
        reasoning_effort: str | None = None,
        client: Any = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._client = client
        self.cache_params = {"max_tokens": max_tokens, "reasoning_effort": reasoning_effort}

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def generate(self, system: str, user: str) -> Generation:
        request: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        return Generation(
            text=(choice.message.content or "").strip(),
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            stop_reason=choice.finish_reason or "",
        )


@register("generator", "echo")
class EchoGenerator:
    """No LLM: reports the start of the top passage. For tests and the offline demo."""

    model = "echo"

    def generate(self, system: str, user: str) -> Generation:
        marker = '<passage id="1">\n'
        if marker not in user:
            return Generation("Offline mode, no LLM called. No passages were retrieved.")
        top = user.split(marker, 1)[1].split("\n</passage>", 1)[0]
        return Generation(
            f"Offline mode, no LLM called. The top passage begins: {top[:200].strip()}",
            stop_reason="end_turn",
        )
