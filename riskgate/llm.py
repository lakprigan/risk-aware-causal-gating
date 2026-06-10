"""LLM provider abstraction for the real-model validation track (paper Sec. 7.1).

This module lets RiskGate drive a *real* LLM as the agent policy instead of the
deterministic MockAgent, while keeping everything else (filters, registry,
tasks, injections, metrics) identical. The model is shown only the
filter-produced visible tool set V_t and must choose one tool via native
function/tool calling; a text-parse fallback covers local models without tool
support.

Design goals:
  * Provider-agnostic: Anthropic, Amazon Bedrock, and any OpenAI-compatible
    endpoint (incl. local Ollama / vLLM) behind one `chat_tool_call` call.
  * Graceful degradation: missing SDKs or API keys raise a clear, catchable
    `LLMUnavailable` so the deterministic track is never affected and the
    runner can skip cleanly.
  * Deterministic-ish: temperature defaults to 0.

No provider SDK is imported at module load; imports are lazy so that simply
importing riskgate never requires `anthropic`, `boto3`, or `openai`.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class LLMUnavailable(RuntimeError):
    """Raised when a provider cannot be used (missing SDK, key, or model)."""


# Substrings that mark a *transient* error worth retrying (throttling, timeouts,
# brief 5xx). Matched case-insensitively against the exception's str().
_TRANSIENT_MARKERS = (
    "throttl", "toomanyrequests", "too many requests", "rate exceeded",
    "rate limit", "429", "503", "serviceunavailable", "service unavailable",
    "timeout", "timed out", "modelnotready", "model not ready",
)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def retry_transient(fn: Callable, *, max_attempts: int = 6,
                    base_delay: float = 1.0, max_delay: float = 30.0):
    """Call fn() with exponential backoff + full jitter on transient errors.

    Used to ride out Bedrock ThrottlingException during long concurrent runs.
    Non-transient errors (bad model id, auth, validation) raise immediately so
    a misconfigured run fails fast instead of retrying pointlessly. After the
    final attempt the last exception propagates so the trial fails loudly.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classify then re-raise
            attempt += 1
            if attempt >= max_attempts or not _is_transient(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            time.sleep(random.uniform(0, delay))  # full jitter


@dataclass
class ToolSpec:
    """A provider-neutral description of one callable tool shown to the model."""
    name: str
    description: str
    # We expose tools as zero-argument calls: the benchmark only needs the
    # model's *choice* of tool, not synthesized arguments (arguments are mocked
    # deterministically by the environment). An empty schema keeps every
    # provider happy.
    parameters: dict = field(default_factory=lambda: {
        "type": "object", "properties": {}, "additionalProperties": False,
    })


@dataclass
class LLMChoice:
    """What the model returned for one step."""
    tool_name: Optional[str]      # chosen tool, or None if it declined / failed to pick
    raw: str = ""                 # raw text (for logging / fallback parsing)
    usage_prompt_tokens: int = 0
    usage_completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------
class LLMProvider:
    """Base class. Subclasses implement `chat_tool_call`."""

    name = "base"

    def __init__(self, model: str, temperature: float = 0.0,
                 max_tokens: int = 256):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat_tool_call(self, system: str, user: str,
                       tools: list[ToolSpec]) -> LLMChoice:
        raise NotImplementedError

    # Shared helper: when a provider returns no tool call, try to recover a
    # tool name from free text (covers local models without tool support).
    @staticmethod
    def _parse_tool_from_text(text: str, tools: list[ToolSpec]) -> Optional[str]:
        if not text:
            return None
        names = [t.name for t in tools]
        # Prefer an exact JSON object {"tool": "..."} if present.
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict):
                cand = obj.get("tool") or obj.get("name") or obj.get("action")
                if cand in names:
                    return cand
        except (json.JSONDecodeError, ValueError):
            pass
        # Otherwise pick the longest tool name that appears as a token.
        lowered = text.lower()
        hits = [n for n in names if n.lower() in lowered]
        if hits:
            return sorted(hits, key=len, reverse=True)[0]
        return None


# ---------------------------------------------------------------------------
# Anthropic (Claude) — native tool use
# ---------------------------------------------------------------------------
class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-sonnet-latest", **kw):
        super().__init__(model, **kw)
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise LLMUnavailable("anthropic SDK not installed (pip install anthropic)") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMUnavailable("ANTHROPIC_API_KEY not set")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()

    def chat_tool_call(self, system, user, tools):
        spec = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        } for t in tools]
        resp = retry_transient(lambda: self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            tools=spec,
            tool_choice={"type": "any"},  # force the model to call exactly one tool
            messages=[{"role": "user", "content": user}],
        ))
        tool_name, text = None, ""
        for block in resp.content:
            if block.type == "tool_use":
                tool_name = block.name
                break
            if block.type == "text":
                text += block.text
        if tool_name is None:
            tool_name = self._parse_tool_from_text(text, tools)
        usage = getattr(resp, "usage", None)
        return LLMChoice(
            tool_name=tool_name, raw=text,
            usage_prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            usage_completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


# ---------------------------------------------------------------------------
# Amazon Bedrock (Converse API) — native tool use across model families
# ---------------------------------------------------------------------------
class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
                 region: Optional[str] = None, **kw):
        super().__init__(model, **kw)
        try:
            import boto3  # noqa: F401
        except ImportError as e:
            raise LLMUnavailable("boto3 not installed (pip install boto3)") from e
        region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        try:
            self._client = boto3.client("bedrock-runtime", region_name=region)
        except Exception as e:  # noqa: BLE001 - surface any credential/config error
            raise LLMUnavailable(f"could not create bedrock-runtime client: {e}") from e

    def chat_tool_call(self, system, user, tools):
        tool_specs = [{
            "toolSpec": {
                "name": t.name,
                "description": t.description,
                "inputSchema": {"json": t.parameters},
            }
        } for t in tools]

        def _infer_config():
            # Some newer models (e.g. Claude Opus 4) reject `temperature` in
            # inferenceConfig ("temperature is deprecated for this model"). We
            # learn that once per provider instance and omit it thereafter.
            cfg = {"maxTokens": self.max_tokens}
            if not getattr(self, "_drop_temperature", False):
                cfg["temperature"] = self.temperature
            return cfg

        def _converse(tool_choice):
            # tool_choice=None => no toolConfig at all (pure text generation).
            # Used as a last resort for models that emit malformed ToolUse blocks
            # under forced tool calling (e.g. some Amazon Nova models); we then
            # recover the chosen tool name from the model's free text.
            kwargs = dict(
                modelId=self.model,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig=_infer_config(),
            )
            if tool_choice is not None:
                kwargs["toolConfig"] = {"tools": tool_specs, "toolChoice": tool_choice}
            return retry_transient(lambda: self._client.converse(**kwargs))

        def _converse_with_temp_fallback(tool_choice):
            try:
                return _converse(tool_choice)
            except Exception as e_t:  # noqa: BLE001
                if "temperature" in str(e_t).lower():
                    # Drop temperature and retry once; remember for next calls.
                    self._drop_temperature = True
                    return _converse(tool_choice)
                raise

        def _is_tooluse_error(e) -> bool:
            s = str(e).lower()
            return "tooluse" in s or "invalid sequence" in s or "ModelErrorException" in str(e)

        try:
            # Prefer forced single-tool choice. Some families (e.g. Meta Llama)
            # reject toolChoice.any; fall back to "auto", which still returns a
            # toolUse block in practice for these models.
            try:
                resp = _converse_with_temp_fallback({"any": {}})
            except Exception as e_any:  # noqa: BLE001
                if _is_tooluse_error(e_any):
                    # Model botched the forced ToolUse (e.g. Nova). Drop tool
                    # forcing entirely and recover the choice from text below.
                    resp = _converse_with_temp_fallback(None)
                elif "toolChoice" in str(e_any) or "ValidationException" in str(e_any):
                    try:
                        resp = _converse_with_temp_fallback({"auto": {}})
                    except Exception as e_auto:  # noqa: BLE001
                        if _is_tooluse_error(e_auto):
                            resp = _converse_with_temp_fallback(None)
                        else:
                            raise
                else:
                    raise
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailable(f"bedrock converse failed: {e}") from e
        tool_name, text = None, ""
        for block in resp["output"]["message"]["content"]:
            if "toolUse" in block:
                tool_name = block["toolUse"]["name"]
                break
            if "text" in block:
                text += block["text"]
        if tool_name is None:
            tool_name = self._parse_tool_from_text(text, tools)
        usage = resp.get("usage", {})
        return LLMChoice(
            tool_name=tool_name, raw=text,
            usage_prompt_tokens=usage.get("inputTokens", 0),
            usage_completion_tokens=usage.get("outputTokens", 0),
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible (OpenAI, Ollama, vLLM, etc.) — function calling
# ---------------------------------------------------------------------------
class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(self, model: str = "gpt-4o-mini",
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None, **kw):
        super().__init__(model, **kw)
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise LLMUnavailable("openai SDK not installed (pip install openai)") from e
        # base_url lets this target local servers (Ollama: http://localhost:11434/v1,
        # vLLM: http://localhost:8000/v1). Local servers often accept any key.
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        api_key = (api_key or os.environ.get("OPENAI_API_KEY")
                   or ("local" if base_url else None))
        if api_key is None:
            raise LLMUnavailable("OPENAI_API_KEY not set (and no base_url for a local server)")
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def chat_tool_call(self, system, user, tools):
        spec = [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        } for t in tools]
        try:
            resp = retry_transient(lambda: self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=spec,
                tool_choice="required",  # force exactly one tool call
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            ))
        except Exception as e:  # noqa: BLE001
            # Some local servers don't support tool_choice="required"; retry "auto".
            try:
                resp = retry_transient(lambda: self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    max_tokens=self.max_tokens, tools=spec, tool_choice="auto",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                ))
            except Exception as e2:  # noqa: BLE001
                raise LLMUnavailable(f"openai-compatible call failed: {e2}") from e2
        msg = resp.choices[0].message
        tool_name, text = None, (msg.content or "")
        if getattr(msg, "tool_calls", None):
            tool_name = msg.tool_calls[0].function.name
        if tool_name is None:
            tool_name = self._parse_tool_from_text(text, tools)
        usage = getattr(resp, "usage", None)
        return LLMChoice(
            tool_name=tool_name, raw=text,
            usage_prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            usage_completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


# ---------------------------------------------------------------------------
# Stub provider — for offline smoke tests and CI (no network, no keys)
# ---------------------------------------------------------------------------
class StubProvider(LLMProvider):
    """A deterministic, no-network provider that emulates a *capable but
    adversarially-susceptible* model: it picks the injection target when that
    target is in the visible tool list (signalled by the runner via the user
    prompt), otherwise it picks the first tool. Useful to validate wiring and
    to demonstrate that ISR depends entirely on whether the target is visible.
    """
    name = "stub"

    def __init__(self, model: str = "stub-v0", **kw):
        super().__init__(model, **kw)

    def chat_tool_call(self, system, user, tools):
        names = [t.name for t in tools]
        # The runner embeds a machine-readable hint line the stub can read so
        # that its behavior is meaningful without a real model. Real providers
        # ignore this line; it is benign natural language to them.
        target = None
        for line in user.splitlines():
            if line.startswith("[INJECTION_TARGET]"):
                target = line.split("]", 1)[1].strip()
        if target and target in names:
            return LLMChoice(tool_name=target, raw=f"obeying:{target}",
                             usage_prompt_tokens=len(user) // 4)
        # Otherwise behave like a benign agent: prefer a gold-ish next tool by
        # taking the first listed tool (the runner lists frontier-first).
        return LLMChoice(tool_name=(names[0] if names else None),
                         raw="benign", usage_prompt_tokens=len(user) // 4)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_provider(spec: str, **kw) -> LLMProvider:
    """Build a provider from a "provider:model" spec.

    Examples:
        "anthropic:claude-3-5-sonnet-latest"
        "bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0"
        "openai_compat:gpt-4o-mini"
        "openai_compat:llama3.1"          (with OPENAI_BASE_URL=...:11434/v1)
        "stub"                            (offline)
    """
    if spec == "stub":
        return StubProvider(**kw)
    if ":" not in spec:
        raise ValueError(f"provider spec must be 'provider:model' or 'stub', got {spec!r}")
    provider, model = spec.split(":", 1)
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicProvider(model=model, **kw)
    if provider == "bedrock":
        return BedrockProvider(model=model, **kw)
    if provider in ("openai", "openai_compat", "openaicompat"):
        return OpenAICompatProvider(model=model, **kw)
    raise ValueError(f"unknown provider {provider!r} in spec {spec!r}")
