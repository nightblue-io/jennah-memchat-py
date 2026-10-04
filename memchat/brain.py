"""The pluggable chat model: Claude through ``anthropic`` (directly or on Amazon
Bedrock), or Gemini through ``google-genai``.

A brain owns the session-local conversation history, so the current chat stays
coherent, and turns the persona, freshly recalled memory and the user's message
into a reply. Long-term memory is Jennah's; nothing Jennah-facing depends on which brain
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
# The same model on Amazon Bedrock. It is a cross-region inference profile id, not
# a bare model id: Bedrock serves current Claude models only through a profile,
# and "global." routes to whichever region has capacity at no premium over a
# single-region profile.
BEDROCK_MODEL = "global.anthropic.claude-sonnet-5-5"
DEFAULT_AWS_REGION = "ap-northeast-1"
# The same id works on AI Studio and on Vertex AI.
GEMINI_MODEL = "gemini-3.8-flash"
MAX_TOKENS = 2048


class Brain(Protocol):
    label: str

    def chat(self, persona: str, recall: str, user_msg: str) -> tuple[str, list[Fact]]:
        """persona is the fixed instruction (the same every turn of a session) and
        recall is this turn's remembered context. They arrive separately because the
        Claude brain must keep the first unchanged and append the second."""
        ...


def use_vertex(env: Mapping[str, str]) -> bool:
    """Whether Gemini goes through Vertex AI rather than an AI Studio key: when
    asked explicitly, or when a GCP project is set and no Studio key is."""
    if env.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
        return True
    return not env.get("GEMINI_API_KEY") and not env.get("GOOGLE_API_KEY") and bool(env.get("GOOGLE_CLOUD_PROJECT"))


def select_provider(provider: str, anthropic_key: str, env: Mapping[str, str]) -> str:
    """Resolve --provider. "auto" prefers Anthropic when an Anthropic key is present,
    else Gemini when a Studio key or Vertex configuration is, so someone with one
    key set needs no flag.

    "bedrock" (Claude on Amazon Bedrock) is never chosen by "auto": AWS credentials
    are present in many shells for reasons that have nothing to do with this demo,
    so having them is no sign of intent."""
    provider = provider.lower()
    if provider == "auto":
        if anthropic_key:
            return "anthropic"
        if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or use_vertex(env):
            return "gemini"
        raise ValueError(
            "no chat credentials found: pass --anthropic-api-key / set ANTHROPIC_API_KEY (Anthropic), "
            "pass --provider bedrock (Claude on Amazon Bedrock, explicit only), "
            "or set the Vertex AI env / GEMINI_API_KEY (Gemini)"
        )
    if provider in ("anthropic", "claude"):
        return "anthropic"
    if provider == "bedrock":
        return "bedrock"
    if provider == "gemini":
        return "gemini"
    raise ValueError(f"unknown --provider {provider!r} (want auto|anthropic|bedrock|gemini)")


def new_brain(provider: str, anthropic_key: str, offer_tool: bool,
              env: Optional[Mapping[str, str]] = None, *,
              aws_region: str = DEFAULT_AWS_REGION, aws_profile: str = "") -> Brain:
    env = os.environ if env is None else env
    chosen = select_provider(provider, anthropic_key, env)
    if chosen == "anthropic":
        return AnthropicBrain(anthropic_key, offer_tool)
    if chosen == "bedrock":
        return bedrock_brain(offer_tool, aws_region, aws_profile)
    return GeminiBrain(offer_tool, env)


def bedrock_brain(offer_tool: bool, region: str, profile: str) -> "AnthropicBrain":
    """Claude on Amazon Bedrock: the Anthropic brain with a Bedrock client, whose
    requests are signed by AWS credentials instead of an Anthropic key.

    The profile is passed explicitly rather than left to AWS_PROFILE because an
    explicitly named profile outranks AWS_ACCESS_KEY_ID in the environment, and
    AWS_PROFILE does not: with bare keys exported, AWS_PROFILE is silently ignored
    and the calls run, without error, in whatever account those keys belong to.
    """
    if not region:
        raise ValueError("--provider bedrock needs an AWS region: pass --aws-region")
    import anthropic

    client = anthropic.AnthropicBedrock(aws_region=region, aws_profile=profile or None)
    return AnthropicBrain("", offer_tool, client=client, model=BEDROCK_MODEL, via="bedrock")


class AnthropicBrain:
    def __init__(self, api_key: str, offer_tool: bool, client=None, *,
                 model: str = ANTHROPIC_MODEL, via: str = "anthropic") -> None:
        if client is None:
            import anthropic

            # An empty key leaves the SDK's own ANTHROPIC_API_KEY lookup in charge.
            client = anthropic.Anthropic(api_key=api_key or None)
        self._client = client
        self._model = model
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
        self.label = f"{via}/{model}"

    def chat(self, persona: str, recall: str, user_msg: str) -> tuple[str, list[Fact]]:
        """persona goes out as the top-level system prompt, which never changes
        during a session, and the turn's recall as a system MESSAGE right after the
        user's.

        Rebuilding the top-level prompt each turn with fresh recall is the obvious
        shape and the wrong one. A thinking block is bound to the exact conversation
        prefix it was produced under, the system prompt included, and an
        organization created on or after 2026-08-31 is refused (400) when an earlier
        block is replayed under a different prefix. So turn 2 of such a session
        failed, on Bedrock and on the direct API alike, while older organizations
        never saw it. Appending recall instead edits nothing that came before: every
        earlier block stays valid, and the unchanged prefix is cacheable as a bonus.

        Old recall stays in the transcript, so a long session carries every snapshot
        (more input tokens per turn). A mid-conversation system message needs a model
        that accepts one: Sonnet 5.5 does; Sonnet 5 answers 400 "role 'system' is not
        supported", so swapping the model to it means moving recall into the user
        turn.
        """
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "system", "content": recall})
        facts: list[Fact] = []
        reply: list[str] = []
        while True:
            kwargs = dict(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=persona,
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

    def chat(self, persona: str, recall: str, user_msg: str) -> tuple[str, list[Fact]]:
        types = self._types
        # The system instruction embeds freshly recalled memory, so it is set per
        # request. Gemini does not bind its history to the instruction it was
        # produced under, so unlike the Claude brain this needs no append-only shape.
        system = persona + "\n\n" + recall
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
