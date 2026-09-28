"""Rendering what a formation or a commit reported.

Each function returns lines as ``(style, text)`` pairs and prints nothing, so the
wording is testable on its own and the caller decides about color. ``style`` is
one of "dim", "cyan" or "yellow".
"""

from __future__ import annotations

from jennah.agent.v1 import memory_pb2

from .authored import triple_text

Line = tuple[str, str]

_DECISIONS = [
    memory_pb2.MEMORY_DECISION_NEW,
    memory_pb2.MEMORY_DECISION_REVISED,
    memory_pb2.MEMORY_DECISION_KNOWN,
    memory_pb2.MEMORY_DECISION_REJECTED,
]


def single_line(s: str) -> str:
    return " ".join(s.split())


def decision_word(d: int) -> str:
    """The short label for a decision, e.g. MEMORY_DECISION_NEW -> "new"."""
    if d == memory_pb2.MEMORY_DECISION_UNSPECIFIED:
        return "?"
    return memory_pb2.MemoryDecision.Name(d).removeprefix("MEMORY_DECISION_").lower()


def candidate_text(c: memory_pb2.FormedCandidate) -> str:
    """A candidate the way it will be remembered: a relationship as its triple,
    anything else as the text that gets stored."""
    if c.kind == memory_pb2.CANDIDATE_KIND_RELATIONSHIP and c.source_entity:
        return triple_text(c.source_entity, c.relationship_type, c.target_entity)
    return single_line(c.text)


def decision_note(c: memory_pb2.FormedCandidate) -> str:
    """The part of a decision that only means something next to its subject: what
    a revision retired, what a known candidate matched, and why a rejected one was
    refused. The rejection reason is how a caller learns that a conversation
    recounting history was not allowed to retire the current fact."""
    if c.decision == memory_pb2.MEMORY_DECISION_REVISED and c.matched_id:
        return f"  (retired {c.matched_id})"
    if c.decision == memory_pb2.MEMORY_DECISION_KNOWN and c.matched_id:
        return f"  (matches {c.matched_id})"
    if c.decision == memory_pb2.MEMORY_DECISION_REJECTED and c.rejection_reason.strip():
        return f"  ({single_line(c.rejection_reason)})"
    return ""


def decision_summary(r: memory_pb2.FormMemoryResponse) -> str:
    """The one-line count. Zero candidates is a legitimate outcome, not a failure:
    a turn can hold nothing worth remembering, and saying so beats "formed: "."""
    if not r.candidates:
        return "nothing worth remembering in that exchange"
    counts: dict[int, int] = {}
    for c in r.candidates:
        counts[c.decision] = counts.get(c.decision, 0) + 1
    return ", ".join(f"{counts[d]} {decision_word(d)}" for d in _DECISIONS if counts.get(d))


def _timestamp(ts, empty: str) -> str:
    if not ts.seconds and not ts.nanos:
        return empty
    return ts.ToDatetime().strftime("%Y-%m-%dT%H:%M:%SZ")


def formation_lines(r: memory_pb2.FormMemoryResponse, verbose: bool) -> list[Line]:
    """What the formation decided.

    A formation receipt reports DECISIONS, because the caller asked for nothing in
    particular and the platform chose. The things a caller cannot infer from row
    counts are printed whether or not --verbose is set: what was retired, what was
    dropped, what was flattened, and what was masked.
    """
    out: list[Line] = []
    if verbose:
        for c in r.candidates:
            out.append(("dim", f"{decision_word(c.decision):<8} {candidate_text(c)}{decision_note(c)}"))
        # A REVISED candidate's replacement is counted as a supersession, not in
        # vector or edge rows, so a turn that only corrected things legitimately
        # reports zero rows. Printing both side by side is the difference between
        # "nothing was stored" and "what was stored replaced something".
        out.append(("dim", (
            f"formed: log={r.execution_log_rows} vec={r.vector_rows}(+{r.chunk_supersessions} superseded) "
            f"nodes={r.graph_node_rows} edges={r.graph_edge_rows}(+{r.edge_supersessions} superseded) "
            f"@ {_timestamp(r.commit_timestamp, 'nothing committed')}"
        )))
    else:
        out.append(("dim", f"[formed: {decision_summary(r)}]"))

    # Retired is not deleted: the old assertion keeps its validity window and stays
    # readable as history. Printed always, because it is the one outcome that
    # changes what the workspace says it knows.
    n = r.edge_supersessions + r.chunk_supersessions
    if n > 0:
        out.append(("cyan", (
            f"[memory] {n} earlier assertion(s) retired by a correction in this turn "
            "(superseded, not overwritten: the previous value stays readable as history)"
        )))
    if r.candidates_dropped > 0:
        out.append(("yellow", (
            f"[memory] {r.candidates_dropped} candidate(s) past the per-formation cap of "
            f"{r.candidate_cap} were dropped, so not everything in that exchange was considered."
        )))
    for s in r.summarized_structures:
        out.append(("yellow", (
            f"[memory] turn {s.turn_index}: {s.description} "
            f"({s.summarized_count} item(s) summarized rather than stored individually)"
        )))
    for rd in r.redactions:
        out.append(("yellow", (
            f"[memory] turn {rd.turn_index}: {rd.masked_count} value(s) masked "
            "before the extraction model saw them"
        )))
    return out


def commit_lines(r: memory_pb2.CommitMemoryResponse, verbose: bool) -> list[Line]:
    """What an authored commit wrote.

    A truncated chunk is printed whether or not --verbose is set: the commit
    succeeded, but the embedding covers only the start of the text, so recall can
    no longer find the turn by anything said in the part that was cut. This demo
    does not set reject_on_truncation, because for a chatbot losing the turn is
    worse than remembering most of it.
    """
    out: list[Line] = []
    if verbose:
        out.append(("dim", (
            f"committed: log={r.execution_log_rows} vec={r.vector_rows} nodes={r.graph_node_rows} "
            f"edges={r.graph_edge_rows} @ {_timestamp(r.commit_timestamp, '?')}"
        )))
    if r.truncated_chunk_ids:
        out.append(("yellow", (
            f"[memory] that message was too long to embed in full ({', '.join(r.truncated_chunk_ids)}). "
            "It is stored, but recall may miss the end of it."
        )))
    return out
