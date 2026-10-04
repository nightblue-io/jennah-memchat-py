"""memchat: a chatbot that remembers across sessions, built on Jennah's Python SDK.

Each turn does recall, answer, form:

1. recall: memory:query for semantic recall of past exchanges (the platform embeds
   the query text), plus memory:inspect to read the knowledge graph back as
   triples.
2. answer: the chat model answers the user, and that is ALL it is asked for. It is
   given no memory tool and no instruction about what to remember.
3. form: memory:form extracts candidate memories from the recent turns, recalls
   what the workspace already holds, reconciles the two, and commits the result
   atomically. The receipt says what it decided about every candidate, and what
   it RETIRED to make room for a correction.

--authored is the other arm: the model is handed a remember_fact tool, and this
client turns the triples it emits into graph nodes and edges itself and writes
them with memory:commit. Both arms write into the same workspace, since formed and
authored memory are the same rows. Run one, then the other, and the difference is
the cognition layer.

Cross-session memory is simply reusing the same workspace id, persisted to a small
state file.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Optional

import grpc

import jennah
from jennah.agent.v1 import memory_pb2

from . import jennah_io
from .authored import STORE_INSTRUCTION, commit_request, rand_id, seed_request
from .brain import DEFAULT_AWS_REGION, Brain, new_brain
from .receipt import commit_lines, formation_lines

# How many recent turns each formation submits: the current exchange plus the two
# before it.
#
# Not just the current exchange, because extraction runs BEFORE recall inside a
# formation, so nothing the workspace already holds can resolve "she", "there" or
# "the second one": only the submitted turns can. The cost of the overlap is that
# already-formed content is re-extracted, which reconciliation reports as KNOWN
# instead of writing twice. That is a token bill, not a correctness problem, and
# it is why the number is small.
FORM_WINDOW = 6

Out = Callable[[str, str], None]

_COLORS = {"dim": "\033[2m", "cyan": "\033[36m", "yellow": "\033[33m"}


def console(stream=None) -> Out:
    """An Out that prints to stdout, colored when stdout is a terminal. "err" goes
    to stderr."""
    stream = stream or sys.stdout
    color = stream.isatty() and not os.environ.get("NO_COLOR")

    def out(style: str, text: str) -> None:
        if style == "err":
            print(text, file=sys.stderr, flush=True)
        elif color and style in _COLORS:
            print(f"  {_COLORS[style]}{text}\033[0m", file=stream, flush=True)
        elif style in _COLORS:
            print(f"  {text}", file=stream, flush=True)
        else:
            print(text, file=stream, flush=True)

    return out


def window(turns: list) -> list:
    """The last FORM_WINDOW turns: the slice each formation submits."""
    return turns[-FORM_WINDOW:]


def formation_key(session_id: str, turn_no: int) -> str:
    """The name of one formation, so that resending it is safe.

    It matters more here than an idempotency key usually does. Extraction is
    nondeterministic, so a blind resend does not repeat the first attempt: it forms
    a second, different set of memory. Under this key a resend replays the original
    receipt and extracts nothing.

    A formation here is (this workspace, this session, this turn ordinal). The
    session id keeps the first turn of every session from being the same formation,
    and the ordinal keeps a user who says "thanks" twice from losing the second one
    to a replay, which hashing the text would do.
    """
    return f"frm_{session_id}_{turn_no}"


def describe(err: BaseException) -> str:
    if isinstance(err, grpc.RpcError) and callable(getattr(err, "code", None)):
        return f"{err.code().name}: {err.details()}"
    return str(err)


def build_persona(authored: bool) -> str:
    """The fixed half of the system prompt: who the model is and what it is for.

    It must come out the same on every turn of a session (the Claude brain sends it
    as a top-level system prompt that may not change mid-conversation), so it reads
    nothing but the arm, fixed at startup. Recalled memory goes in build_recall.

    The instruction about STORING memory appears only in the authored arm, and its
    absence by default is not a simplification: a prompt telling the model what to
    remember while the platform independently decides the same thing would be two
    extractors with one workspace, disagreeing at the caller's expense.
    """
    parts = [
        "You are Memo, a warm, concise assistant with long-term memory that persists across sessions. "
        "Each user message is followed by what you remember that is relevant to it. Personalize using "
        "that remembered context and refer back to it naturally; the most recent one is current. "
    ]
    if authored:
        parts.append(STORE_INSTRUCTION)
    return "".join(parts)


def build_recall(rec: jennah_io.Recall) -> str:
    """What Jennah recalled for this turn."""
    parts = ["# What you already know (knowledge graph)\n"]
    if rec.facts:
        parts.extend(f"- {f}\n" for f in rec.facts)
    else:
        parts.append("(nothing yet, this may be your first conversation)\n")
    parts.append("\n# Relevant snippets from past conversations\n")
    if rec.snippets:
        parts.extend(f"- {s.text}\n" for s in rec.snippets)
    else:
        parts.append("(none retrieved)\n")
    return "".join(parts)


def retired_note(n: int) -> str:
    return f" (+{n} retired, not shown)" if n else ""


class Chat:
    """One session: recall, answer, then write, per line of input.

    transcript is the formation arm's own copy of the recent turns, in the wire
    type memory:form takes. The brain's history cannot serve: it is in whichever
    vendor SDK's message type answered.
    """

    def __init__(self, client: jennah.Client, brain: Brain, agent_id: str, *,
                 authored: bool, verbose: bool, out: Out) -> None:
        self.client = client
        self.brain = brain
        self.agent_id = agent_id
        self.authored = authored
        self.verbose = verbose
        self.out = out
        self.transcript: list[memory_pb2.ConversationTurn] = []
        self.session_id = rand_id("sess")
        self.turn_no = 0

    def _emit(self, lines) -> None:
        for style, text in lines:
            self.out(style, text)

    def turn(self, line: str) -> None:
        try:
            rec = jennah_io.recall(self.client, self.agent_id, line)
        except grpc.RpcError as e:
            self.out("err", f"error: recall: {describe(e)}")
            return
        summary = f"recalled {len(rec.facts)} fact(s){retired_note(rec.retired)}, {len(rec.snippets)} past snippet(s)"
        if self.verbose:
            self.out("dim", summary + ":")
            for f in rec.facts:
                self.out("dim", f"  - {f}")
            for s in rec.snippets:
                self.out("dim", f"  ~ {s.text}{s.prov}")
        else:
            self.out("dim", f"[{summary}]")

        try:
            reply, facts = self.brain.chat(build_persona(self.authored), build_recall(rec), line)
        except Exception as e:  # the vendor SDKs raise their own types
            self.out("err", f"error: chat model: {e}")
            return
        self.out("plain", f"\nmemo> {reply}")

        # The reply is on screen BEFORE memory is written. A formation runs model
        # inference and a retrieval before it writes, so it is a seconds-class call
        # by contract, and making the user wait on it to read an answer the model
        # already produced would be self-inflicted latency.
        self.turn_no += 1
        self.transcript += [
            memory_pb2.ConversationTurn(role=memory_pb2.TURN_ROLE_USER, content=line),
            memory_pb2.ConversationTurn(role=memory_pb2.TURN_ROLE_ASSISTANT, content=reply),
        ]
        if self.authored:
            self._commit(line, reply, facts)
        else:
            self._form()

    def _form(self) -> None:
        turns = window(self.transcript)
        key = formation_key(self.session_id, self.turn_no)
        if self.verbose:
            self.out("dim", f"forming memory from {len(turns)} turn(s), key {key} ...")
        else:
            self.out("dim", "[forming memory ...]")
        try:
            resp = jennah_io.form(self.client, self.agent_id, turns, key)
        except grpc.RpcError as e:
            self.out("err", f"warning: could not form this turn's memory: {describe(e)}")
            return
        self._emit(formation_lines(resp, self.verbose))

    def _commit(self, line: str, reply: str, facts) -> None:
        req, stored = commit_request(self.agent_id, line, reply, facts)
        try:
            resp = jennah_io.commit(self.client, req)
        except grpc.RpcError as e:
            self.out("err", f"warning: could not persist this turn's memory: {describe(e)}")
            return
        if self.verbose:
            for s in stored:
                self.out("dim", f"stored fact: {s}")
        self._emit(commit_lines(resp, self.verbose))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="memchat",
        description="A chatbot that remembers across sessions, with memory kept in Jennah.",
    )
    p.add_argument("--endpoint", default=None,
                   help=f"Jennah gRPC endpoint as host:port (default {jennah.DEFAULT_ENDPOINT})")
    p.add_argument("--insecure", action="store_true",
                   help="connect without TLS, for a local plaintext server")
    p.add_argument("--state", default="memchat-state.json",
                   help="path to the local state file holding the workspace id (default %(default)s)")
    p.add_argument("--agent", default="",
                   help="use this EXISTING agent workspace instead of the one in the state file, for a "
                        "workspace provisioned out of band (e.g. one with a vocabulary declared on it). "
                        "Never creates and never writes the state file")
    p.add_argument("--provider", default="auto",
                   help="chat model: auto|anthropic|bedrock|gemini (auto prefers Anthropic, else Gemini, "
                        "by which credentials are set; bedrock is Claude on Amazon Bedrock and is never "
                        "picked by auto)")
    p.add_argument("--aws-region", default=DEFAULT_AWS_REGION,
                   help="AWS region for --provider bedrock")
    p.add_argument("--aws-profile", default="",
                   help="AWS named profile for --provider bedrock; empty uses the default credential chain. "
                        "Prefer this over AWS_PROFILE, which exported AWS_ACCESS_KEY_ID silently overrides")
    # The region is not a secret, so its env var may be the default.
    p.add_argument("--region", default=os.environ.get("JENNAH_REGION", ""),
                   help="home region for a NEW workspace (e.g. us-central1), also read from $JENNAH_REGION; "
                        "empty uses the platform default. List regions with 'jnh agents regions'")
    # Secrets are never flag defaults, so --help cannot print them. The SDK and the
    # vendor SDKs read their env vars themselves when a flag is not passed.
    p.add_argument("--jennah-api-key", default="",
                   help="Jennah API key (jennah_sk_...); falls back to $JENNAH_API_KEY, then 'jnh login'")
    p.add_argument("--anthropic-api-key", default="",
                   help="Anthropic API key; falls back to $ANTHROPIC_API_KEY")
    p.add_argument("--verbose", action="store_true",
                   help="print recalled memory and full receipts each turn")
    p.add_argument("--authored", action="store_true",
                   help="extract and author the memory writes in this client (remember_fact + memory:commit) "
                        "instead of letting memory:form do it. Writes into the same workspace")
    return p.parse_args(argv)


def connect(args: argparse.Namespace) -> jennah.Client:
    """The SDK client, with the credential resolved the SDK's way: the flag when
    given, else $JENNAH_API_KEY, else the 'jnh login' session."""
    try:
        return jennah.Client(
            api_key=args.jennah_api_key.strip() or None,
            endpoint=args.endpoint,
            insecure=args.insecure,
        )
    except jennah.NoCredentialError:
        raise jennah_io.StartupError(
            "no Jennah credential found: pass --jennah-api-key, set JENNAH_API_KEY, or run 'jnh login'"
        ) from None
    except jennah.JennahError as e:
        raise jennah_io.StartupError(str(e)) from None


def main(argv: Optional[list[str]] = None, out: Optional[Out] = None) -> int:
    args = parse_args(argv)
    out = out or console()
    try:
        # The credential is checked first, so a missing one fails before anything
        # touches the network.
        client = connect(args)
        anthropic_key = args.anthropic_api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
        try:
            brain = new_brain(args.provider, anthropic_key, args.authored,
                              aws_region=args.aws_region, aws_profile=args.aws_profile)
        except ValueError as e:
            raise jennah_io.StartupError(str(e)) from None
        out("plain", f"chat model: {brain.label}")
        out("plain", "memory: authored in this client (remember_fact + memory:commit)" if args.authored
            else "memory: formed by Jennah (memory:form)")
        try:
            agent_id, where = jennah_io.resolve_workspace(
                client, agent=args.agent, state_path=args.state, region=args.region)
        except grpc.RpcError as e:
            raise jennah_io.StartupError(f"workspace: {describe(e)}") from None
        out("plain", where)
        if args.authored:
            try:
                jennah_io.commit(client, seed_request(agent_id))
            except grpc.RpcError as e:
                raise jennah_io.StartupError(f"seed user node: {describe(e)}") from None
        else:
            out("plain", f"vocabulary: {jennah_io.vocabulary_summary(client, agent_id)}")
    except jennah_io.StartupError as e:
        out("err", f"memchat: {e}")
        return 1

    out("plain", "\nmemchat: a chatbot that remembers across sessions (Ctrl-D or /exit to quit).")
    out("plain", "Tell it about yourself, quit, run it again, and it'll recall.")
    chat = Chat(client, brain, agent_id, authored=args.authored, verbose=args.verbose, out=out)
    try:
        while True:
            try:
                line = input("\nyou> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line in ("/exit", "/quit"):
                break
            chat.turn(line)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
    out("plain", "\nbye. Your memory is saved in Jennah.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
