import datetime
import json
import types

import grpc
import pytest
from google.protobuf import timestamp_pb2

import jennah
from jennah import credentials
from jennah.agent.v1 import memory_pb2 as m

from fake import API_KEY
from memchat import __main__ as app
from memchat import jennah_io
from memchat.authored import Fact, commit_request, node_id, norm_rel, seed_request
from memchat.brain import AnthropicBrain, GeminiBrain, bedrock_brain, select_provider
from memchat.receipt import commit_lines, formation_lines


class Events:
    """An Out that records (style, text), and a shared timeline tests can append to."""

    def __init__(self):
        self.lines = []
        self.timeline = []

    def __call__(self, style, text):
        self.lines.append((style, text))
        self.timeline.append(("out", text))

    def text(self):
        return "\n".join(t for _, t in self.lines)


class FakeBrain:
    label = "fake/brain"

    def __init__(self, reply="ok", facts=None):
        self.reply = reply
        self.facts = facts or []
        self.systems = []
        self.personas = []

    def chat(self, persona, recall, user_msg):
        self.personas.append(persona)
        self.systems.append(persona + "\n\n" + recall)
        return self.reply, list(self.facts)


def ts(dt):
    t = timestamp_pb2.Timestamp()
    t.FromDatetime(dt)
    return t


def chat(client, fake, *, authored=False, verbose=False, brain=None):
    fake.agents.add("demo.a")
    ev = Events()
    return app.Chat(client, brain or FakeBrain(), "demo.a", authored=authored, verbose=verbose, out=ev), ev


# ---- 2.1 flags: secrets stay out of --help ----


def test_help_does_not_leak_the_key(monkeypatch, capsys):
    secret = "jennah_sk_" + "Q7xS3cr3tV4lu3Zz9"
    monkeypatch.setenv("JENNAH_API_KEY", secret)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-An0th3rS3cr3t")
    with pytest.raises(SystemExit):
        app.parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "--jennah-api-key" in help_text and "--authored" in help_text
    for part in ("Q7xS3cr3tV4lu3Zz9", "S3cr3t", "An0th3r"):
        assert part not in help_text


# ---- 2.2 credentials ----


def test_no_credential_exits_before_any_call(machine, fake):
    ev = Events()
    code = app.main(["--endpoint", fake.endpoint, "--insecure", "--provider", "gemini"], out=ev)
    assert code == 1
    err = ev.text()
    assert "--jennah-api-key" in err and "JENNAH_API_KEY" in err and "jnh login" in err
    assert fake.calls == []


def test_signed_in_session_is_used(machine, fake):
    credentials.save_session(credentials.Session(
        endpoint="https://jennah.alphaus.cloud", access_token=API_KEY, token_type="Bearer"))
    args = app.parse_args(["--endpoint", fake.endpoint, "--insecure"])
    client = app.connect(args)
    try:
        assert client.credential.origin is jennah.Origin.FILE
        fake.agents.add("demo.s")
        jennah_io.require_agent(client, "demo.s")
    finally:
        client.close()


# ---- 2.3 workspace resolution never guesses ----


def test_first_run_creates_later_runs_reuse(tmp_path, fake, client):
    state = str(tmp_path / "state.json")
    first, msg = jennah_io.resolve_workspace(client, agent="", state_path=state, region="asia-northeast1")
    assert first.startswith("demo.") and "created" in msg and "asia-northeast1" in msg
    created = fake.of("CreateAgent")
    assert len(created) == 1 and created[0].request.region == "asia-northeast1"
    assert json.load(open(state)) == {"agent_id": first}

    second, msg = jennah_io.resolve_workspace(client, agent="", state_path=state, region="us-central1")
    assert second == first and "reusing" in msg
    assert len(fake.of("CreateAgent")) == 1


def test_mistyped_agent_fails_naming_both_causes(tmp_path, fake, client):
    with pytest.raises(jennah_io.StartupError) as e:
        jennah_io.resolve_workspace(client, agent="demo.typo", state_path=str(tmp_path / "s.json"), region="")
    msg = str(e.value)
    assert "demo.typo" in msg and "not there" in msg and "cannot reach" in msg
    assert fake.of("CreateAgent") == []
    assert not (tmp_path / "s.json").exists()


def test_agent_flag_leaves_state_file_alone(tmp_path, fake, client):
    state = tmp_path / "s.json"
    state.write_text('{"agent_id": "demo.mine"}')
    before = state.read_bytes()
    fake.agents.add("demo.operator")
    got, msg = jennah_io.resolve_workspace(client, agent="demo.operator", state_path=str(state), region="x")
    assert got == "demo.operator" and "untouched" in msg
    assert state.read_bytes() == before
    assert fake.of("CreateAgent") == []


# ---- 2.4 recall ----


def test_recall_pages_and_filters_retired_edges(fake, client):
    now = datetime.datetime.now(datetime.timezone.utc)
    page1 = m.InspectMemoryResponse(
        graph=m.GraphInspectResult(
            nodes=[m.GraphNode(node_id="n1", label="Chew")],
            edges=[
                m.GraphEdge(edge_id="e1", source_node_id="n1", target_node_id="n2", relationship_type="LIVES_IN",
                            invalid_at=ts(now - datetime.timedelta(days=1))),
                m.GraphEdge(edge_id="e2", source_node_id="n1", target_node_id="n3", relationship_type="LIVES_IN"),
            ]),
        next_node_token="more-nodes",
    )
    page2 = m.InspectMemoryResponse(graph=m.GraphInspectResult(
        nodes=[m.GraphNode(node_id="n2", label="Osaka"), m.GraphNode(node_id="n3", label="Tokyo")],
        edges=[m.GraphEdge(edge_id="e3", source_node_id="n1", target_node_id="n2", relationship_type="VISITED",
                           invalid_at=ts(now + datetime.timedelta(days=1)))]))
    fake.inspect_pages = [page1, page2]

    facts, retired = jennah_io.recall_facts(client, "demo.a")
    assert facts == ["Chew lives in Tokyo", "Chew visited Osaka"]
    assert retired == 1
    reqs = [c.request for c in fake.of("InspectMemory")]
    assert len(reqs) == 2
    assert reqs[0].graph.node_limit == 200 and reqs[0].graph.edge_limit == 200
    assert reqs[1].graph.node_page_token == "more-nodes" and reqs[1].graph.edge_page_token == ""


def test_recall_stops_at_ten_pages(fake, client):
    fake.inspect_pages = [m.InspectMemoryResponse(next_edge_token="again")]
    jennah_io.recall_facts(client, "demo.a")
    assert len(fake.of("InspectMemory")) == 10


def test_retired_count_is_shown(fake, client):
    now = datetime.datetime.now(datetime.timezone.utc)
    fake.inspect_pages = [m.InspectMemoryResponse(graph=m.GraphInspectResult(edges=[
        m.GraphEdge(edge_id="e1", source_node_id="a", target_node_id="b", relationship_type="R",
                    invalid_at=ts(now - datetime.timedelta(seconds=5)))]))]
    c, ev = chat(client, fake)
    c.turn("hi")
    assert "[recalled 0 fact(s) (+1 retired, not shown), 0 past snippet(s)]" in ev.text()


def test_provenance_is_verbose_only_and_never_in_the_prompt(fake, client):
    fake.query = m.QueryMemoryResponse(semantic=m.SemanticResult(matches=[m.SemanticMatch(
        chunk_id="c1", raw_content="User likes   hiking",
        metadata={"jennah.source_step": "step_formed_42", "jennah.source_turns": "0-1"})]))
    brain = FakeBrain()
    c, ev = chat(client, fake, brain=brain, verbose=True)
    c.turn("what do I like?")
    assert "- User likes hiking" in brain.systems[0]
    assert "step_formed_42" not in brain.systems[0] and "formed by" not in brain.systems[0]
    assert "~ User likes hiking  [formed by step_formed_42, turn(s) 0-1]" in ev.text()

    c2, ev2 = chat(client, fake)
    c2.turn("again")
    assert "step_formed_42" not in ev2.text()


def test_semantic_query_limit(fake, client):
    jennah_io.recall_semantic(client, "demo.a", "hello")
    req = fake.of("QueryMemory")[0].request
    assert req.semantic.query_text == "hello" and req.semantic.limit == 6


# ---- 2.5 vocabulary banner ----


def test_vocabulary_counts(fake, client):
    fake.vocabulary = m.GetMemoryVocabularyResponse(resolved=m.MemoryVocabulary(
        entity_classes=[m.EntityClass(), m.EntityClass()], relation_types=[m.RelationType()]))
    assert jennah_io.vocabulary_summary(client, "demo.a") == "2 entity class(es), 1 relation type(s) in effect"
    assert fake.of("GetMemoryVocabulary")[0].request.scope_id == "demo.a"


def test_vocabulary_none_declared(fake, client):
    got = jennah_io.vocabulary_summary(client, "demo.a")
    assert got.startswith("none declared") and "jnh vocabulary declare --scope demo.a" in got


def test_vocabulary_permission_denied_is_not_fatal(tmp_path, fake):
    fake.vocabulary = grpc.StatusCode.PERMISSION_DENIED
    got = jennah_io.vocabulary_summary(
        jennah.Client(endpoint=fake.endpoint, insecure=True, api_key=API_KEY), "demo.a")
    assert "agent.vocabulary:read" in got and "not readable" in got


def test_startup_survives_denied_vocabulary(tmp_path, fake, monkeypatch):
    fake.vocabulary = grpc.StatusCode.PERMISSION_DENIED
    monkeypatch.setattr(app, "new_brain", lambda *a, **k: FakeBrain())
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    ev = Events()
    code = app.main(["--endpoint", fake.endpoint, "--insecure", "--jennah-api-key", API_KEY,
                     "--state", str(tmp_path / "s.json")], out=ev)
    assert code == 0
    assert "vocabulary: not readable" in ev.text() and "bye." in ev.text()


# ---- 3.1 brain selection, and no tools by default ----


def test_auto_provider_selection():
    assert select_provider("auto", "sk-ant-x", {}) == "anthropic"
    assert select_provider("auto", "sk-ant-x", {"GEMINI_API_KEY": "g"}) == "anthropic"
    assert select_provider("auto", "", {"GEMINI_API_KEY": "g"}) == "gemini"
    assert select_provider("auto", "", {"GOOGLE_API_KEY": "g"}) == "gemini"
    assert select_provider("auto", "", {"GOOGLE_CLOUD_PROJECT": "jennah-hq"}) == "gemini"
    assert select_provider("auto", "", {"GOOGLE_GENAI_USE_VERTEXAI": "true"}) == "gemini"
    with pytest.raises(ValueError, match="no chat credentials"):
        select_provider("auto", "", {})
    with pytest.raises(ValueError, match="unknown --provider"):
        select_provider("openai", "", {})


def test_bedrock_is_explicit_only():
    assert select_provider("bedrock", "", {}) == "bedrock"
    # AWS credentials in the environment are no sign of intent, so auto never picks it.
    aws = {"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "s", "AWS_PROFILE": "p"}
    with pytest.raises(ValueError, match="no chat credentials"):
        select_provider("auto", "", aws)
    with pytest.raises(ValueError, match="needs an AWS region"):
        bedrock_brain(False, "", "p")


class _AnthropicStub:
    def __init__(self, responses):
        self.requests = []
        self._responses = list(responses)
        self.messages = self

    def create(self, **kwargs):
        # history is mutated after the call, so keep the messages as sent
        self.requests.append(dict(kwargs, messages=list(kwargs["messages"])))
        return self._responses.pop(0)


def _text_resp(text):
    return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)], stop_reason="end_turn")


def test_anthropic_default_arm_sends_no_tools():
    stub = _AnthropicStub([_text_resp("hello")])
    b = AnthropicBrain("", offer_tool=False, client=stub)
    assert b.chat("persona", "recall", "hi") == ("hello", [])
    assert "tools" not in stub.requests[0]
    assert stub.requests[0]["system"] == "persona"


def test_anthropic_recall_is_appended_never_edited():
    # The persona stays the top-level system prompt and each turn's recall is a
    # system message after the user's, so every request extends the last one
    # instead of rewriting it (earlier thinking blocks are bound to that prefix).
    stub = _AnthropicStub([_text_resp("one"), _text_resp("two")])
    b = AnthropicBrain("", offer_tool=False, client=stub)
    b.chat("persona", "recall 1", "hi")
    b.chat("persona", "recall 2", "again")
    first, second = stub.requests
    assert first["messages"] == [{"role": "user", "content": "hi"}, {"role": "system", "content": "recall 1"}]
    assert second["system"] == first["system"] == "persona"
    assert second["messages"][:2] == first["messages"]
    assert second["messages"][3:] == [{"role": "user", "content": "again"}, {"role": "system", "content": "recall 2"}]


def test_anthropic_authored_arm_collects_facts():
    tool = types.SimpleNamespace(type="tool_use", id="t1", input={"relationship": "lives in", "object": "Tokyo"})
    stub = _AnthropicStub([
        types.SimpleNamespace(content=[tool], stop_reason="tool_use"),
        _text_resp("noted!"),
    ])
    b = AnthropicBrain("", offer_tool=True, client=stub)
    reply, facts = b.chat("persona", "recall", "I live in Tokyo")
    assert reply == "noted!" and facts == [Fact("", "lives in", "Tokyo")]
    assert stub.requests[0]["tools"][0]["name"] == "remember_fact"
    assert stub.requests[1]["messages"][-1]["content"][0]["tool_use_id"] == "t1"


class _GeminiStub:
    def __init__(self, text):
        from google.genai import types as gt

        self.configs = []
        self._text = text
        self._gt = gt
        self.models = self

    def generate_content(self, model, contents, config):
        self.configs.append(config)
        gt = self._gt
        return gt.GenerateContentResponse(candidates=[gt.Candidate(
            content=gt.Content(role="model", parts=[gt.Part(text=self._text)]))])


def test_gemini_default_arm_sends_no_tools():
    stub = _GeminiStub("hi there")
    b = GeminiBrain(False, {"GEMINI_API_KEY": "k"}, client=stub)
    assert b.chat("persona", "recall", "hi") == ("hi there", [])
    assert not stub.configs[0].tools
    assert stub.configs[0].system_instruction == "persona\n\nrecall"
    assert b.label.endswith("(ai-studio)")


def test_gemini_authored_arm_offers_remember_fact():
    stub = _GeminiStub("ok")
    GeminiBrain(True, {"GEMINI_API_KEY": "k"}, client=stub).chat("persona", "recall", "hi")
    decl = stub.configs[0].tools[0].function_declarations[0]
    assert decl.name == "remember_fact"
    assert set(decl.parameters.properties) == {"subject", "relationship", "object"}
    assert decl.parameters.required == ["relationship", "object"]


def test_gemini_vertex_label():
    b = GeminiBrain(False, {"GOOGLE_CLOUD_PROJECT": "jennah-hq"}, client=_GeminiStub("x"))
    assert b.label == "gemini/gemini-3.8-flash (vertex:jennah-hq/global)"


def test_default_prompt_has_no_store_instruction(fake, client):
    brain = FakeBrain()
    c, _ = chat(client, fake, brain=brain)
    c.turn("hi")
    assert "remember_fact" not in brain.systems[0] and "store" not in brain.systems[0].lower()


def test_persona_is_fixed_across_turns(fake, client):
    brain = FakeBrain()
    c, _ = chat(client, fake, brain=brain)
    c.turn("hi")
    fake.query = m.QueryMemoryResponse(semantic=m.SemanticResult(matches=[m.SemanticMatch(
        chunk_id="c1", raw_content="User likes hiking")]))
    c.turn("again")
    assert brain.personas[0] == brain.personas[1]
    assert "User likes hiking" not in brain.personas[1] and "- User likes hiking" in brain.systems[1]


# ---- 3.2 the turn loop ----


def test_window_is_the_last_six_turns(fake, client):
    c, _ = chat(client, fake)
    for i in range(6):
        c.brain.reply = f"reply {i}"
        c.turn(f"message {i}")
    last = fake.of("FormMemory")[-1].request
    assert len(last.turns) == 6
    assert [t.content for t in last.turns] == [
        "message 3", "reply 3", "message 4", "reply 4", "message 5", "reply 5"]
    assert last.turns[0].role == m.TURN_ROLE_USER and last.turns[1].role == m.TURN_ROLE_ASSISTANT
    assert len(fake.of("FormMemory")[0].request.turns) == 2


def test_same_text_different_turns_get_different_keys(fake, client):
    c, _ = chat(client, fake)
    c.turn("thanks")
    c.turn("thanks")
    keys = [call.request.formation_key for call in fake.of("FormMemory")]
    assert len(set(keys)) == 2
    assert keys[0] == f"frm_{c.session_id}_1" and keys[1] == f"frm_{c.session_id}_2"
    assert fake.extractions == 2


def test_resent_formation_replays(fake, client):
    turns = [m.ConversationTurn(role=m.TURN_ROLE_USER, content="x")]
    fake.form = lambda req: m.FormMemoryResponse(vector_rows=1)
    a = jennah_io.form(client, "demo.a", turns, "frm_s_1")
    b = jennah_io.form(client, "demo.a", turns, "frm_s_1")
    assert a == b and fake.extractions == 1


def test_deadlines(fake, client):
    c, _ = chat(client, fake)
    c.turn("hi")
    form = fake.of("FormMemory")[0].deadline
    assert 290 < form <= 302  # gRPC rounds the wire deadline up
    for method in ("QueryMemory", "InspectMemory"):
        d = fake.of(method)[0].deadline
        assert 0 < d <= 62


def test_commit_deadline_is_ordinary(fake, client):
    c, _ = chat(client, fake, authored=True)
    c.turn("hi")
    assert 0 < fake.of("CommitMemory")[0].deadline <= 62


def test_reply_is_printed_before_formation(fake, client):
    c, ev = chat(client, fake, brain=FakeBrain(reply="the answer"))

    def form(req):
        ev.timeline.append(("form", req.formation_key))
        return m.FormMemoryResponse()

    fake.form = form
    c.turn("question")
    kinds = [k for k, t in ev.timeline if k == "form" or "memo> the answer" in t]
    assert kinds == ["out", "form"]


def test_formation_failure_is_a_warning(fake, client):
    fake.fail["FormMemory"] = grpc.StatusCode.UNAVAILABLE
    c, ev = chat(client, fake)
    c.turn("hi")
    assert any(s == "err" and "could not form this turn's memory" in t for s, t in ev.lines)


# ---- 3.3 receipt rendering ----


def cand(decision, text="", **kw):
    return m.FormedCandidate(decision=decision, text=text, kind=m.CANDIDATE_KIND_FACT, **kw)


def texts(lines):
    return [t for _, t in lines]


def test_receipt_counts():
    r = m.FormMemoryResponse(candidates=[
        cand(m.MEMORY_DECISION_NEW), cand(m.MEMORY_DECISION_NEW), cand(m.MEMORY_DECISION_REVISED),
        cand(m.MEMORY_DECISION_REJECTED)])
    assert texts(formation_lines(r, False)) == ["[formed: 2 new, 1 revised, 1 rejected]"]


def test_receipt_nothing_worth_remembering():
    assert texts(formation_lines(m.FormMemoryResponse(), False)) == [
        "[formed: nothing worth remembering in that exchange]"]


def test_receipt_supersession_without_verbose():
    r = m.FormMemoryResponse(candidates=[cand(m.MEMORY_DECISION_REVISED)], edge_supersessions=1)
    assert formation_lines(r, False)[1] == (
        "cyan",
        "[memory] 1 earlier assertion(s) retired by a correction in this turn "
        "(superseded, not overwritten: the previous value stays readable as history)")


def test_receipt_dropped_summarized_redacted():
    r = m.FormMemoryResponse(
        candidates=[cand(m.MEMORY_DECISION_NEW)], candidates_dropped=3, candidate_cap=20,
        summarized_structures=[m.SummarizedStructure(turn_index=2, description="a table of prices",
                                                     summarized_count=14)],
        redactions=[m.RedactionRecord(turn_index=0, masked_count=2)])
    assert texts(formation_lines(r, False))[1:] == [
        "[memory] 3 candidate(s) past the per-formation cap of 20 were dropped, "
        "so not everything in that exchange was considered.",
        "[memory] turn 2: a table of prices (14 item(s) summarized rather than stored individually)",
        "[memory] turn 0: 2 value(s) masked before the extraction model saw them",
    ]


def test_receipt_verbose_notes():
    r = m.FormMemoryResponse(
        candidates=[
            m.FormedCandidate(decision=m.MEMORY_DECISION_REVISED, kind=m.CANDIDATE_KIND_RELATIONSHIP,
                              source_entity="Chew", relationship_type="LIVES_IN", target_entity="Tokyo",
                              matched_id="e_old"),
            cand(m.MEMORY_DECISION_KNOWN, "likes  hiking", matched_id="c_1"),
            cand(m.MEMORY_DECISION_REJECTED, "used to live in Osaka", rejection_reason="recounts history"),
            cand(m.MEMORY_DECISION_NEW, "has a cat"),
        ],
        vector_rows=1, graph_edge_rows=0, edge_supersessions=1, execution_log_rows=1)
    got = texts(formation_lines(r, True))
    assert got[:4] == [
        "revised  Chew lives in Tokyo  (retired e_old)",
        "known    likes hiking  (matches c_1)",
        "rejected used to live in Osaka  (recounts history)",
        "new      has a cat",
    ]
    assert got[4] == ("formed: log=1 vec=1(+0 superseded) nodes=0 edges=0(+1 superseded) "
                      "@ nothing committed")
    assert got[5].startswith("[memory] 1 earlier assertion(s) retired")


# ---- 4.1 authored arm: label convergence ----


def test_two_phrasings_converge_on_one_edge():
    a, _ = commit_request("demo.a", "u", "r", [Fact("Chew", "is CTO of", "NightBlue")])
    b, _ = commit_request("demo.a", "u", "r", [Fact("NightBlue", "has CTO", "Chew")])
    ea, eb = a.graph.edges[0], b.graph.edges[0]
    assert ea.edge_id == eb.edge_id
    assert (ea.source_node_id, ea.relationship_type, ea.target_node_id) == (
        node_id("NightBlue"), "HAS_CTO", node_id("Chew"))


def test_user_references_fold_onto_the_anchor():
    for ref in ("", "me", "I", "the user", " User ", "myself"):
        assert node_id(ref) == "user"
    assert node_id("Chew") == node_id("  chew ")
    assert norm_rel("is named") == "IS_NAMED" and norm_rel("  ") == "RELATED_TO"
    req, stored = commit_request("demo.a", "u", "r", [Fact("", "lives in", "Tokyo")])
    assert [n.node_id for n in req.graph.nodes] == [node_id("Tokyo")]
    assert stored == ["user lives in Tokyo"]


def test_seed_is_the_user_anchor():
    req = seed_request("demo.a")
    assert [(n.node_id, n.label) for n in req.graph.nodes] == [("user", "User")]


def test_authored_start_seeds_the_anchor(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(app, "new_brain", lambda *a, **k: FakeBrain())
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    ev = Events()
    assert app.main(["--endpoint", fake.endpoint, "--insecure", "--jennah-api-key", API_KEY, "--authored",
                     "--state", str(tmp_path / "s.json")], out=ev) == 0
    seeds = fake.of("CommitMemory")
    assert len(seeds) == 1 and seeds[0].request.graph.nodes[0].node_id == "user"
    assert fake.of("GetMemoryVocabulary") == []


# ---- 4.2 commitTurn ----


def test_one_commit_carries_all_three_sections_and_truncation_is_disclosed(fake, client):
    fake.commit = m.CommitMemoryResponse(truncated_chunk_ids=["chunk_big"])
    facts = [Fact("", "lives in", "Tokyo"), Fact("", "lives in", "Tokyo"), Fact("Chew", "is CTO of", "NightBlue")]
    c, ev = chat(client, fake, authored=True, brain=FakeBrain(reply="got it", facts=facts))
    c.turn("I live in Tokyo, and Chew is CTO of NightBlue")
    commits = fake.of("CommitMemory")
    assert len(commits) == 1
    req = commits[0].request
    assert req.log.tool_used == "memchat"
    assert req.vectors[0].raw_content == "User: I live in Tokyo, and Chew is CTO of NightBlue\nAssistant: got it"
    assert len(req.graph.edges) == 2  # the repeated fact is deduplicated within the commit
    assert {n.label for n in req.graph.nodes} == {"Tokyo", "Chew", "NightBlue"}
    assert fake.of("FormMemory") == []
    assert ("yellow", "[memory] that message was too long to embed in full (chunk_big). "
                      "It is stored, but recall may miss the end of it.") in ev.lines


def test_commit_receipt_quiet_without_truncation():
    assert commit_lines(m.CommitMemoryResponse(vector_rows=1), False) == []
