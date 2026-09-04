"""Language-model providers, behind one small interface.

The advisor does not care which model it is talking to, because the model's job
is deliberately tiny: turn a sentence into a typed plan, and write prose over an
evidence ledger. Nothing here can compute, decide legality, or reach the kernel.

Swapping providers is therefore a configuration change, not an architectural
one -- which is the point. Two are wired up:

  NVIDIA NIM   OpenAI-compatible, https://integrate.api.nvidia.com/v1
  Anthropic    the Messages API

With neither key present the advisor runs its deterministic lane and still
answers the entire shipped question set.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

_ENV_LOADED = False


def load_env(path: str = ".env") -> None:
    """Minimal .env loader -- no dependency, and it never overwrites a real
    environment variable, so CI and shell exports always win."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (path, os.path.join(root, path)):
        if not os.path.isfile(candidate):
            continue
        with open(candidate, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return


DEFAULT_MODELS = {
    "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",
    "anthropic": "claude-sonnet-5",
}


@dataclass
class LLM:
    provider: str
    model: str
    _client: Any

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def complete(self, system: str, user: str, max_tokens: int = 600,
                 temperature: float = 0.0) -> str:
        """One turn, no tools, no history. Returns raw text.

        Deliberately the narrowest possible surface: the model is never handed a
        tool it could call, so it cannot reach the kernel on its own.
        """
        if self.provider == "anthropic":
            msg = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                temperature=temperature,
                messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in msg.content
                           if getattr(b, "type", None) == "text").strip()

        # OpenAI-compatible (NVIDIA NIM)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.provider == "nvidia":
            # Reasoning traces are pure latency here: this model is a parser and
            # a narrator, and both jobs are single-step.
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}}
        r = self._client.chat.completions.create(**kwargs)
        return (r.choices[0].message.content or "").strip()


def get_client(prefer: str | None = None) -> LLM | None:
    """Return a configured provider, or None to run deterministic-only."""
    load_env()
    if os.environ.get("CREWOPS_NO_MODEL"):
        return None

    order = [prefer] if prefer else ["nvidia", "anthropic"]
    for provider in order:
        if provider == "nvidia" and os.environ.get("NVIDIA_API_KEY"):
            try:
                from openai import OpenAI
            except ImportError:
                continue
            model = os.environ.get("CREWOPS_MODEL") or DEFAULT_MODELS["nvidia"]
            if "/" not in model:
                model = DEFAULT_MODELS["nvidia"]
            return LLM("nvidia", model, OpenAI(
                base_url=os.environ.get(
                    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                api_key=os.environ["NVIDIA_API_KEY"],
                timeout=float(os.environ.get("CREWOPS_TIMEOUT", "12")),
                max_retries=0))

        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
            except ImportError:
                continue
            model = os.environ.get("CREWOPS_MODEL") or DEFAULT_MODELS["anthropic"]
            if model.startswith("nvidia/"):
                model = DEFAULT_MODELS["anthropic"]
            return LLM("anthropic", model, anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                timeout=float(os.environ.get("CREWOPS_TIMEOUT", "12")),
                max_retries=0))
    return None
