"""The pluggable chat model: Claude through ``anthropic``, or Gemini through ``google-genai``.

A brain owns the session-local conversation history, so the current chat stays
coherent, and turns a freshly built system prompt plus the user's message into a
reply. Long-term memory is Jennah's; nothing Jennah-facing depends on which brain
answered.

The facts a brain returns belong to the --authored arm alone. By default what is
worth remembering is the platform's decision, so the model is offered no tool and
the tool-call branches below never fire. That is the shape of the change
formation makes to a client: not a different call, one fewer job.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Protocol

from .authored import TOOL_DESC, TOOL_NAME, TOOL_PROPERTIES, TOOL_REQUIRED, Fact, fact_from_args

ANTHROPIC_MODEL = "claude-sonnet-5-5"
# The same id works on AI Studio and on Vertex AI.
GEMINI_MODEL = "gemini-3.8-flash"
MAX_TOKENS = 2048


class Brain(Protocol):
    label: str

    def chat(self, system: str, user_msg: str) -> tuple[str, list[Fact]]: ...


def use_vertex(env: Mapping[str, str]) -> bool:
    """Whether Gemini goes through Vertex AI rather than an AI Studio key: when
    asked explicitly, or when a GCP project is set and no Studio key is."""
    if env.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
        return True
    return not env.get("GEMINI_API_KEY") and not env.get("GOOGLE_API_KEY") and bool(env.get("GOOGLE_CLOUD_PROJECT"))


def select_provider(provider: str, anthropic_key: str, env: Mapping[str, str]) -> str:
    """Resolve --provider. "auto" prefers Anthropic when an Anthropic key is present,
    else Gemini when a Studio key or Vertex configuration is, so someone with one
    key set needs no flag."""
    provider = provider.lower()
    if provider == "auto":
        if anthropic_key:
            return "anthropic"
        if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or use_vertex(env):
            return "gemini"
        raise ValueError(
            "no chat credentials found: set GEMINI_API_KEY or the Vertex AI env (Gemini), "
            "or pass --anthropic-api-key / set ANTHROPIC_API_KEY (Anthropic), or pass --provider"
        )
    if provider in ("anthropic", "claude"):
        return "anthropic"
    if provider == "gemini":
        return "gemini"
    raise ValueError(f"unknown --provider {provider!r} (want auto|gemini|anthropic)")


def new_brain(provider: str, anthropic_key: str, offer_tool: bool,
              env: Optional[Mapping[str, str]] = None) -> Brain:
    env = os.environ if env is None else env
    if select_provider(provider, anthropic_key, env) == "anthropic":
        return AnthropicBrain(anthropic_key, offer_tool)
    return GeminiBrain(offer_tool, env)


class AnthropicBrain:
    def __init__(self, api_key: str, offer_tool: bool, client=None) -> None:
        if client is None:
            import anthropic

            # An empty key leaves the SDK's own ANTHROPIC_API_KEY lookup in charge.
            client = anthropic.Anthropic(api_key=api_key or None)
        self._client = client
        self._history: list[dict] = []
        self._tools = []
        if offer_tool:
            self._tools = [{
                "name": TOOL_NAME,
                "description": TOOL_DESC,
                "input_schema": {
                    "type": "object",
                    "properties": {k: {"type": "string", "description": d} for k, d in TOOL_PROPERTIES.items()},
                    "required": TOOL_REQUIRED,
                },
            }]
        self.label = f"anthropic/{ANTHROPIC_MODEL}"

    def chat(self, system: str, user_msg: str) -> tuple[str, list[Fact]]:
        self._history.append({"role": "user", "content": user_msg})
        facts: list[Fact] = []
        reply: list[str] = []
        while True:
            kwargs = dict(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=self._history,
                thinking={"type": "adaptive"},
            )
            if self._tools:
                kwargs["tools"] = self._tools
            resp = self._client.messages.create(**kwargs)
            # The whole content goes back into history, thinking blocks included,
            # because a tool-use turn must be continued with them intact.
            self._history.append({"role": "assistant", "content": resp.content})
            results = []
            for blk in resp.content:
                if blk.type == "text":
                    reply.append(blk.text)
                elif blk.type == "tool_use":
                    f = fact_from_args(dict(blk.input or {}))
                    if f:
                        facts.append(f)
                    results.append({"type": "tool_result", "tool_use_id": blk.id, "content": "noted"})
            if resp.stop_reason != "tool_use":
                break
            self._history.append({"role": "user", "content": results})
        return "".join(reply).strip(), facts


class GeminiBrain:
    def __init__(self, offer_tool: bool, env: Mapping[str, str], client=None) -> None:
        from google.genai import types

        self._types = types
        backend = "ai-studio"
        if use_vertex(env):
            # Vertex uses Application Default Credentials, not an API key
            # (gcloud auth application-default login).
            project = env.get("GOOGLE_CLOUD_PROJECT", "")
            if not project:
                raise ValueError(
                    "Vertex AI selected but GOOGLE_CLOUD_PROJECT is not set "
                    "(and run: gcloud auth application-default login)"
                )
            # Gemini 3.8 is served on the global endpoint.
            location = env.get("GOOGLE_CLOUD_LOCATION") or env.get("GOOGLE_CLOUD_REGION") or "global"
            backend = f"vertex:{project}/{location}"
            if client is None:
                from google import genai

                client = genai.Client(vertexai=True, project=project, location=location)
        elif client is None:
            from google import genai

            client = genai.Client(api_key=env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"))
        self._client = client
        self._history: list = []
        self._tools = None
        if offer_tool:
            self._tools = [types.Tool(function_declarations=[types.FunctionDeclaration(
                name=TOOL_NAME,
                description=TOOL_DESC,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        k: types.Schema(type=types.Type.STRING, description=d) for k, d in TOOL_PROPERTIES.items()
                    },
                    required=TOOL_REQUIRED,
                ),
            )])]
        self.label = f"gemini/{GEMINI_MODEL} ({backend})"

    def chat(self, system: str, user_msg: str) -> tuple[str, list[Fact]]:
        types = self._types
        # The system prompt embeds freshly recalled memory, so it is set per request.
        # Automatic function calling is off because the loop below handles
        # remember_fact itself.
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=MAX_TOKENS,
            tools=self._tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self._history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_msg)]))
        facts: list[Fact] = []
        reply: list[str] = []
        while True:
            resp = self._client.models.generate_content(model=GEMINI_MODEL, contents=self._history, config=config)
            if not resp.candidates or resp.candidates[0].content is None:
                break
            content = resp.candidates[0].content
            self._history.append(content)
            results = []
            for part in content.parts or []:
                if part.function_call is not None:
                    fc = part.function_call
                    f = fact_from_args(dict(fc.args or {}))
                    if f:
                        facts.append(f)
                    results.append(types.Part.from_function_response(name=fc.name, response={"result": "noted"}))
                elif part.text and not part.thought:
                    reply.append(part.text)
            if not results:
                break
            # Feed the tool results back so the model can produce its final reply.
            self._history.append(types.Content(role="user", parts=results))
        return "".join(reply).strip(), facts
