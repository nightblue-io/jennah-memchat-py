"""Every Jennah call memchat makes, through the SDK's synchronous Client.

Nothing here builds a request to the platform by hand or opens a channel of its
own: each call is a generated stub on ``client.agents`` or ``client.memory``, so
credential resolution, session renewal and retry classification are the SDK's.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
from typing import Optional

import grpc

from jennah import Client
from jennah.agent.v1 import agent_pb2, memory_pb2

from .authored import rand_id, triple_text

# CALL_TIMEOUT is the deadline for the millisecond-class calls (query, inspect,
# commit, workspace checks). FORM_TIMEOUT is for memory:form alone, and it is five
# times longer on purpose.
#
# A formation is slow by contract, not by accident: it runs two bounded generative
# calls plus a retrieval before it writes, and the platform checks at startup that
# the whole thing fits the 300 s its load balancer allows a request. A 60 s client
# deadline would abandon formations the server goes on to finish and commit, and
# report memory that WAS written as a failure. The formation key makes that
# recoverable, but recovery should not be the ordinary path.
CALL_TIMEOUT = 60.0
FORM_TIMEOUT = 300.0

# Every workspace this demo creates is named under "demo.". '.' is the platform's
# agent-selector separator and selector matching is segment-anchored, so one role
# selector "demo.*" reaches every id minted here and nothing else. That lets a demo
# run be scoped to a throwaway role instead of needing blanket agent access.
DEMO_PREFIX = "demo."

# The graph read-out's page size, and a bound on the walk so a runaway workspace
# cannot stall a chat turn.
RECALL_PAGE_LIMIT = 200
MAX_RECALL_PAGES = 10

SEMANTIC_LIMIT = 6


class StartupError(Exception):
    """A condition that stops the demo before the first prompt."""


# ---- workspace resolution ----


def load_state(path: str) -> str:
    """The workspace id in the state file, or "" when there is none yet."""
    try:
        with open(path, encoding="utf-8") as f:
            return str(json.load(f).get("agent_id") or "")
    except FileNotFoundError:
        return ""


def save_state(path: str, agent_id: str) -> None:
    """Persist the workspace id, and only that. Nothing else needs tracking: graph
    writes are idempotent upserts on caller-supplied ids, so re-asserting a fact
    across sessions converges instead of duplicating."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"agent_id": agent_id}, f, indent=2)


def require_agent(client: Client, agent_id: str) -> None:
    """Check that a workspace named with --agent is really there, so a mistyped id
    fails at startup naming itself rather than three lines into a conversation.

    A not-found is indistinguishable from a workspace this credential cannot
    reach, because the platform collapses the two on purpose (a refusal that
    confirmed the id exists would be a disclosure). So the message names both.
    """
    try:
        client.agents.GetAgent(agent_pb2.GetAgentRequest(agent_instance_id=agent_id), timeout=CALL_TIMEOUT)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise StartupError(
                f"agent workspace {agent_id!r} is not there, or this credential cannot reach it "
                f"(the platform answers both the same way). Create it with 'jnh agents create {agent_id}', "
                "or drop --agent to let this demo mint its own"
            ) from None
        raise StartupError(f"check agent workspace {agent_id!r}: {e.details()}") from None


def create_agent(client: Client, region: str) -> str:
    """Provision a workspace. region is applied only here, because an agent is
    pinned to one home region for its lifetime; "" means the platform default."""
    wanted = DEMO_PREFIX + rand_id("memchat")
    resp = client.agents.CreateAgent(
        agent_pb2.CreateAgentRequest(agent_instance_id=wanted, agent_name="memchat-demo", region=region),
        timeout=CALL_TIMEOUT,
    )
    return resp.agent.agent_instance_id or wanted


def resolve_workspace(client: Client, *, agent: str, state_path: str, region: str) -> tuple[str, str]:
    """The workspace this run uses, and the startup line that says where it came from.

    --agent names a workspace someone else provisioned, which is the ordinary way
    to run against one with a vocabulary declared on it. It never creates and never
    touches the state file. Creating would let a mistyped id silently mint a second,
    empty workspace, and the demo would then report remembering nothing, which
    reads as a platform fault. Writing the id to the state file would leave later
    flagless runs pointed at the operator's workspace.
    """
    agent = agent.strip()
    if agent:
        require_agent(client, agent)
        return agent, f"using agent workspace {agent} (--agent; the state file is untouched)"

    agent_id = load_state(state_path)
    if agent_id:
        return agent_id, f"reusing agent workspace {agent_id} (memory carries over)"
    agent_id = create_agent(client, region)
    save_state(state_path, agent_id)
    where = f"region {region}" if region else "platform default region"
    return agent_id, f"created agent workspace {agent_id} ({where})"


# ---- recall ----


@dataclasses.dataclass
class Snippet:
    """One recalled chunk: the text for the prompt, and separately where it came
    from. Provenance is for the person watching --verbose; splicing it into the
    prompt would put ids in front of the model that it has no use for."""

    text: str
    prov: str = ""


@dataclasses.dataclass
class Recall:
    facts: list[str]
    snippets: list[Snippet]
    retired: int


def provenance(metadata) -> str:
    """Where a recalled chunk came from, when a formation wrote it.

    Formation stamps what it writes under reserved jennah.* keys a caller cannot
    set itself. An authored chunk carries none, so this is "" for it."""
    step = metadata.get("jennah.source_step", "")
    if not step:
        return ""
    turns = metadata.get("jennah.source_turns", "")
    return f"  [formed by {step}, turn(s) {turns}]" if turns else f"  [formed by {step}]"


def recall_semantic(client: Client, agent_id: str, query: str) -> list[Snippet]:
    resp = client.memory.QueryMemory(
        memory_pb2.QueryMemoryRequest(
            agent_instance_id=agent_id,
            semantic=memory_pb2.SemanticQuery(query_text=query, limit=SEMANTIC_LIMIT),
        ),
        timeout=CALL_TIMEOUT,
    )
    out = []
    for m in resp.semantic.matches:
        content = m.raw_content.strip()
        if content:
            out.append(Snippet(" ".join(content.split()), provenance(m.metadata)))
    return out


def is_retired(edge: memory_pb2.GraphEdge, now: datetime.datetime) -> bool:
    """Whether an edge's valid-time window has closed as of now. An unset
    invalid_at means still current, and a future one is still live."""
    if not edge.HasField("invalid_at"):
        return False
    return edge.invalid_at.ToDatetime(tzinfo=datetime.timezone.utc) <= now


def recall_facts(client: Client, agent_id: str, now: Optional[datetime.datetime] = None) -> tuple[list[str], int]:
    """The whole knowledge graph read back as triples, and how many retired edges
    were left out.

    This uses memory:inspect rather than a traversal because of edge DIRECTION. A
    traversal row does not project an edge's endpoints, so orientation is only
    known when a step pins a direction, and which way a fact points is the model's
    phrasing choice: an outgoing walk from any one node silently loses half the
    graph. Inspect returns every edge with its source and target.

    Retired edges are filtered here because inspect enumerates what is STORED,
    superseded assertions included, where a query answers what is TRUE. Left in, a
    correction would leave the prompt holding both "lives in Osaka" and "lives in
    Tokyo" as current. The retired edge is still stored and readable, which is what
    makes it history rather than a deletion; the count lets the caller say so.
    """
    now = now or datetime.datetime.now(datetime.timezone.utc)
    labels: dict[str, str] = {}
    edges: list[memory_pb2.GraphEdge] = []
    node_tok = edge_tok = ""
    for _ in range(MAX_RECALL_PAGES):
        resp = client.memory.InspectMemory(
            memory_pb2.InspectMemoryRequest(
                agent_instance_id=agent_id,
                graph=memory_pb2.InspectGraph(
                    node_limit=RECALL_PAGE_LIMIT,
                    edge_limit=RECALL_PAGE_LIMIT,
                    node_page_token=node_tok,
                    edge_page_token=edge_tok,
                ),
            ),
            timeout=CALL_TIMEOUT,
        )
        for n in resp.graph.nodes:
            labels[n.node_id] = n.label
        edges.extend(resp.graph.edges)
        # The two listings exhaust independently, so keep going while EITHER has
        # more. An empty token means that listing is done, not merely this page.
        node_tok, edge_tok = resp.next_node_token, resp.next_edge_token
        if not node_tok and not edge_tok:
            break

    # A node id with no label means the node listing was cut short while its edges
    # came back; the raw id is more useful to the model than dropping the fact.
    def label(nid: str) -> str:
        return labels.get(nid, "").strip() or nid

    facts: list[str] = []
    seen: set[str] = set()
    retired = 0
    for e in edges:
        if is_retired(e, now):
            retired += 1
            continue
        line = triple_text(label(e.source_node_id), e.relationship_type, label(e.target_node_id))
        if line not in seen:
            facts.append(line)
            seen.add(line)
    return facts, retired


def recall(client: Client, agent_id: str, query: str) -> Recall:
    snippets = recall_semantic(client, agent_id, query)
    facts, retired = recall_facts(client, agent_id)
    return Recall(facts, snippets, retired)


# ---- vocabulary ----


def vocabulary_summary(client: Client, agent_id: str) -> str:
    """The vocabulary formation will classify this workspace's memory against, for
    the startup banner.

    READ ONLY, and not because it was simpler. Declaring a vocabulary is
    management-class, so an operator does it out of band and the agent lives with
    what resolves. Even the read needs agent.vocabulary:read, which the member
    default bundle does not carry, so a refusal is reported as the ordinary outcome
    it is. A chatbot that refused to start over it would be making the wrong thing
    essential.
    """
    try:
        resp = client.memory.GetMemoryVocabulary(
            memory_pb2.GetMemoryVocabularyRequest(scope_id=agent_id), timeout=CALL_TIMEOUT
        )
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:
            return (
                "not readable with this credential (it lacks the agent.vocabulary:read permission); "
                "formation still classifies against whatever is declared"
            )
        return f"could not be read ({e.details()}); formation still classifies against whatever is declared"
    # RESOLVED, not the declaration: this scope's own declaration if it has one and
    # the enterprise default otherwise, which is what a formation classifies against.
    v = resp.resolved
    classes, relations = len(v.entity_classes), len(v.relation_types)
    if not classes and not relations:
        return (
            "none declared, so entities are extracted untyped "
            f"(declare one with: jnh vocabulary declare --scope {agent_id} --from-file vocabulary.yaml)"
        )
    return f"{classes} entity class(es), {relations} relation type(s) in effect"


# ---- writes ----


def form(client: Client, agent_id: str, turns: list, key: str) -> memory_pb2.FormMemoryResponse:
    """Hand the recent turns to memory:form.

    This is the whole write path of the default arm: no extraction, no node ids,
    no direction convention, no normalizer, because the platform runs extract,
    recall, reconcile and commit behind this one call.

    observed_at is left unset, meaning "when this formation is received", which
    for a live conversation is when it happened. It exists for backfills, where
    valid time is not ingest time.
    """
    return client.memory.FormMemory(
        memory_pb2.FormMemoryRequest(scope_id=agent_id, turns=turns, formation_key=key),
        timeout=FORM_TIMEOUT,
    )


def commit(client: Client, req: memory_pb2.CommitMemoryRequest) -> memory_pb2.CommitMemoryResponse:
    return client.memory.CommitMemory(req, timeout=CALL_TIMEOUT)
