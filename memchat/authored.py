"""The --authored arm: memory extracted in this client and written with memory:commit.

Everything here is what memory:form replaces. It is kept, behind --authored,
because the contrast is the most useful thing this demo can show an integrator:
this is the apparatus you own the moment you decide to extract memory in your own
client (node ids, a direction convention, a relationship normalizer), and the
default arm owns none of it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import secrets

from jennah.agent.v1 import memory_pb2

# The authored arm's anchor: the stable node its graph is reachable from. It is
# the one node with a fixed id rather than a content hash.
#
# It is a CLIENT CONVENTION, not a platform concept, which is why the default arm
# neither seeds nor uses it. Formation extracts relationships between entities the
# conversation NAMES, so "my name is Chew" becomes facts about an entity called
# Chew and nothing links them to the person typing. The authored arm gets away
# with anchoring only because it owns the tool schema and can rule that an omitted
# subject means the user.
USER_NODE = "user"

# The remember_fact tool, described once and mapped into each chat SDK's tool type
# by brain.py so the backends stay in lockstep. It stores ONE
# (subject)-[relationship]->(object) triple per call.
#
# The subject is a field, not an assumption. A tool that hardwired the user as the
# subject made every fact a spoke off one hub, and the model packed anything else
# into the two strings it had ("Gucci, Haruka, Suna-kun" as one object). Hence the
# emphasis on exactly one entity per field.
TOOL_NAME = "remember_fact"
TOOL_DESC = (
    "Store ONE durable fact in long-term memory as a (subject)-[relationship]->(object) triple. "
    "Call once per fact, and call as many times as a message needs: a fact mentioning several "
    "entities is several calls, never one call with a list crammed into a field. Use for stable "
    "facts worth recalling in future sessions (the user's name, preferences, job, location and "
    "goals, and the people, organizations, teams and things they tell you about, including how "
    "those relate to each other); do NOT store transient chit-chat or questions."
)
TOOL_PROPERTIES = {
    "subject": (
        "the single entity the fact is about, e.g. 'Alice', 'NightBlue', 'FinOps Consulting'. "
        "OMIT it whenever the fact is the user talking about themselves ('my name is X', 'I live "
        "in Y', 'I work at Z'), and keep omitting it once you know their name: the user already "
        "has a dedicated node, so naming them here creates a duplicate of them. Name a subject "
        "only for facts about someone or something else."
    ),
    "relationship": (
        "short verb phrase linking subject to object, e.g. 'is named', 'likes', 'lives in', "
        "'works at', 'has cto', 'owns', 'reports to', 'has member'"
    ),
    "object": (
        "the single entity or value the relationship points at, e.g. 'Alice', 'hiking', 'Tokyo', "
        "'NightBlue'. Exactly one, never a list: three members of a team is three calls, and two "
        "roles held by one person is two calls."
    ),
}
# subject is optional: omitted means the user, which keeps the common case ("my
# name is Sabrina") a two-field call.
TOOL_REQUIRED = ["relationship", "object"]

# The instruction the authored arm adds to the system prompt. The default arm has
# no equivalent, see buildSystemPrompt in __main__.
STORE_INSTRUCTION = (
    "Whenever the user shares a durable fact, call remember_fact to store it as one "
    "(subject, relationship, object) triple: their name, preferences, job, location and goals, "
    "and also the people, organizations and teams they mention and how those relate to one "
    "another. One call per fact, one entity per field: a team with three members is three "
    "calls, not one call listing three names. Do not store transient chit-chat."
)


@dataclasses.dataclass(frozen=True)
class Fact:
    """One (subject)-[relationship]->(object) triple the model chose to store. An
    empty subject means the user."""

    subj: str
    rel: str
    obj: str


def fact_from_args(args: dict) -> Fact | None:
    """A tool call's arguments as a Fact, or None when a required field is blank."""
    rel = str(args.get("relationship") or "")
    obj = str(args.get("object") or "")
    if not rel.strip() or not obj.strip():
        return None
    return Fact(str(args.get("subject") or ""), rel, obj)


# INVERSE_REL canonicalizes edge DIRECTION. A key is a relationship the model emits
# pointing the "wrong" way; its value is the canonical relationship to store once
# source and target are swapped. So "Chew is CTO of NightBlue" and "NightBlue has
# CTO Chew" both land as NightBlue -[HAS_CTO]-> Chew, one edge with one id.
#
# The convention is container first: the organization or owner is the source, and
# the person or part it contains is the target.
#
# This is NOT an ontology. A predicate absent from this table is stored exactly as
# the model phrased it. The table lists only pairs observed being emitted BOTH ways
# across repeated runs of the same conversation.
INVERSE_REL = {
    "IS_CEO_OF": "HAS_CEO",
    "IS_CTO_OF": "HAS_CTO",
    "IS_COO_OF": "HAS_COO",
    "IS_CFO_OF": "HAS_CFO",
    "IS_MEMBER_OF": "HAS_MEMBER",
    "IS_PART_OF": "HAS_PART",
    "IS_A_DEPARTMENT_OF": "HAS_DEPARTMENT",
    "IS_OWNED_BY": "OWNS",
    "BELONGS_TO": "HAS_MEMBER",
}


def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def rand_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def is_user_ref(label: str) -> bool:
    return label.strip().lower() in ("", "user", "the user", "me", "i", "myself")


def node_id(label: str) -> str:
    """The stable node id for an entity label, content-hashed so the same entity
    named twice converges on one node. Anything that names the user folds onto the
    fixed anchor, so a phrasing choice cannot strand facts on a rival node."""
    if is_user_ref(label):
        return USER_NODE
    return "n_" + short_hash(label.strip().lower())


def subject_label(subj: str) -> str:
    return "user" if is_user_ref(subj) else subj.strip()


def norm_rel(s: str) -> str:
    """A verb phrase as an edge RelationshipType, e.g. "is named" -> "IS_NAMED"."""
    out = "".join(c if c.isascii() and c.isalnum() else "_" for c in s.strip().upper()).strip("_")
    return out or "RELATED_TO"


def pretty_rel(s: str) -> str:
    """A stored RelationshipType back as a readable phrase."""
    return s.replace("_", " ").lower() if s else "->"


def triple_text(subj: str, rel: str, obj: str) -> str:
    return f"{subj} {pretty_rel(rel)} {obj}"


def seed_request(agent_id: str) -> memory_pb2.CommitMemoryRequest:
    """The user anchor, written on every authored start.

    Every start rather than once at bootstrap, because the arm can change between
    runs: a workspace created by the default arm has no anchor, and an authored
    commit naming an absent node is rejected. The write is an idempotent upsert of
    a fixed label, so re-sending it cannot drift.
    """
    return memory_pb2.CommitMemoryRequest(
        agent_instance_id=agent_id,
        graph=memory_pb2.GraphWrite(nodes=[memory_pb2.GraphNode(node_id=USER_NODE, label="User")]),
    )


def commit_request(agent_id: str, user_msg: str, reply: str, facts: list[Fact]):
    """One turn as a single CommitMemory: the exchange as a vector chunk, a log
    step, and the facts as graph nodes and edges, written atomically.

    Returns the request and the facts as readable triples, for --verbose.
    """
    nodes: list[memory_pb2.GraphNode] = []
    edges: list[memory_pb2.GraphEdge] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    def add_node(label: str) -> str:
        nid = node_id(label)
        # The anchor carries its own label from the seed; re-writing it here would
        # overwrite "User" with whatever the model typed.
        if nid != USER_NODE and nid not in seen_nodes:
            nodes.append(memory_pb2.GraphNode(node_id=nid, label=label.strip()))
            seen_nodes.add(nid)
        return nid

    stored: list[str] = []
    for f in facts:
        # Both endpoints get a node whichever way the edge ends up pointing, so the
        # flip below only reorients the edge.
        src, dst = add_node(f.subj), add_node(f.obj)
        src_label, dst_label = subject_label(f.subj), f.obj.strip()
        rel = norm_rel(f.rel)
        if rel in INVERSE_REL:
            src, dst = dst, src
            src_label, dst_label = dst_label, src_label
            rel = INVERSE_REL[rel]
        # Keyed on the canonical ids and the normalized relationship, so the same
        # fact phrased differently converges on one edge.
        eid = "e_" + short_hash(f"{src}|{rel}|{dst}")
        # A mutation set cannot carry two writes for the same key, so dedup within
        # this commit. Idempotency across commits is the server's job.
        if eid not in seen_edges:
            edges.append(memory_pb2.GraphEdge(
                edge_id=eid, source_node_id=src, target_node_id=dst, relationship_type=rel))
            seen_edges.add(eid)
        stored.append(triple_text(src_label, rel, dst_label))

    req = memory_pb2.CommitMemoryRequest(
        agent_instance_id=agent_id,
        log=memory_pb2.ExecutionLogStep(
            step_id=rand_id("step"),
            thought_process="conversation turn",
            tool_used="memchat",
            tool_input=user_msg[:500],
            tool_output=reply[:1000],
        ),
        vectors=[memory_pb2.VectorChunk(
            chunk_id=rand_id("chunk"),
            raw_content=f"User: {user_msg}\nAssistant: {reply}",
        )],
    )
    if nodes or edges:
        req.graph.CopyFrom(memory_pb2.GraphWrite(nodes=nodes, edges=edges))
    return req, stored
