## memchat (Python)

A command-line chatbot that remembers you across sessions. Tell it about
yourself, quit, run it again, and it recalls what you said. Memory is kept in
Jennah, and every Jennah call goes through the Python SDK (`jennah-sdk-py`) over
gRPC.

By default the chat model only answers. Deciding what is worth remembering is
Jennah's job: after each reply, memchat submits the recent turns to
`memory:form`, which extracts candidate memories, compares them with what the
workspace already holds, and commits the result in one write. The receipt
reports what it decided about every candidate, including which earlier facts a
correction retired.

`--authored` switches to client-side memory: the chat model gets a
`remember_fact` tool, and memchat turns the triples it emits into graph nodes
and edges itself and writes them with `memory:commit`. Both modes write to the
same workspace, so you can run one and then the other and compare.

## Requirements

- Python 3.10 or later
- A Jennah credential: a `jennah_sk_` API key, or a session from `jnh login`
- A chat model: an Anthropic API key, Claude on Amazon Bedrock through an AWS
  profile allowed to call `bedrock:InvokeModel` on the
  `global.anthropic.claude-sonnet-5-5` inference profile, Gemini on Vertex AI
  through Application Default Credentials, or a Gemini (AI Studio) API key

## Install

```sh
pip install git+https://github.com/nightblue-io/jennah-memchat-py
```

or from a clone:

```sh
git clone https://github.com/nightblue-io/jennah-memchat-py
cd jennah-memchat-py
pip install .
```

## Run

```sh
# Jennah: an API key, or skip this and run `jnh login`
export JENNAH_API_KEY=jennah_sk_...

# Chat model: pick one
export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic
export GOOGLE_CLOUD_PROJECT=my-proj   # Gemini on Vertex AI
export GEMINI_API_KEY=...             # Gemini on AI Studio

memchat
```

With no `--provider`, memchat uses Anthropic when its key is set, otherwise
Gemini. On Vertex AI the location defaults to `global`.

Claude on Amazon Bedrock is never picked automatically. Choose it and name the
AWS profile:

```sh
# --aws-region defaults to ap-northeast-1
memchat --provider bedrock --aws-profile my-profile
```

Pass the profile with `--aws-profile` rather than `AWS_PROFILE`: when
`AWS_ACCESS_KEY_ID` is also exported, it silently takes precedence over
`AWS_PROFILE`, and the calls run in that key's account instead.

`python -m memchat` works the same way from a clone.

The first run creates a workspace named `demo.memchat_<random>` and saves its
id to `memchat-state.json`. Later runs reuse it, which is all it takes for
memory to carry over between sessions. Type `/exit` or press Ctrl-D to quit.

```text
you> Hi! I'm Chew and I live in Osaka. I'm CTO of a company
called NightBlue.

memo> Nice to meet you, Chew! ...
  [forming memory ...]
  [formed: 4 new]

you> Quick correction: I moved to Tokyo last month.

memo> Got it, thanks for the update! ...
  [forming memory ...]
  [formed: 2 revised, 2 known]
  [memory] 2 earlier assertion(s) retired by a correction in
  this turn (superseded, not overwritten: the previous value
  stays readable as history)
```

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--endpoint` | `jennah-grpc.alphaus.cloud:443` | Jennah gRPC endpoint as `host:port` |
| `--insecure` | off | Connect without TLS, for a local plaintext server |
| `--state` | `memchat-state.json` | File holding the workspace id |
| `--agent` | | Use this existing workspace instead of the state file. Never creates one and never writes the state file |
| `--provider` | `auto` | `auto`, `anthropic`, `bedrock` or `gemini`. `auto` picks Anthropic if its key is set, otherwise Gemini, and never Bedrock |
| `--aws-region` | `ap-northeast-1` | AWS region for `--provider bedrock` |
| `--aws-profile` | | AWS named profile for `--provider bedrock`. Empty uses the default credential chain |
| `--region` | `$JENNAH_REGION` | Home region for a new workspace. Only applied at creation |
| `--jennah-api-key` | | Jennah API key. Falls back to `$JENNAH_API_KEY`, then the `jnh login` session |
| `--anthropic-api-key` | | Anthropic key. Falls back to `$ANTHROPIC_API_KEY` |
| `--verbose` | off | Print recalled memory and every candidate's decision each turn |
| `--authored` | off | Extract memory in this client (`remember_fact` + `memory:commit`) |

Secrets are never flag defaults, so `--help` does not print them.

## How a turn works

1. **Recall.** A semantic `memory:query` (limit 6) finds relevant past
   snippets, and `memory:inspect` reads the knowledge graph back as triples.
   Edges whose validity has ended (retired by a correction) are left out of the
   prompt and counted, as in `(+1 retired, not shown)`.
2. **Answer.** The chat model gets a fixed persona plus that recall. In the
   default mode it has no tools and no instruction about what to remember.
   Claude gets the persona as a system prompt that never changes during a
   session, and each turn's recall as a `role: "system"` message after the
   user's message. Rebuilding the system prompt every turn would invalidate
   earlier thinking blocks, which newer Anthropic organizations reject with a
   400 on the second turn. Older recall stays in the transcript, so long
   sessions use more input tokens.
3. **Form.** The reply is printed first, then the last 6 turns go to
   `memory:form`. Each formation carries a key, `frm_<session>_<turn>`, so a
   retry replays the original receipt instead of extracting twice, and saying
   the same thing on two turns still gives two formations.

Formation runs model inference before it writes, so it takes seconds. memchat
allows it 300 seconds and gives every other call 60.

Some receipt lines print even without `--verbose`, because they report something
you would otherwise not learn: retired facts, candidates dropped past the
per-formation cap, structures summarized instead of stored item by item, and
values masked before extraction. In `--authored` mode, a message too long to
embed in full is reported the same way.

## Running against a prepared workspace

`--agent` is for a workspace someone else set up, for example one with a
vocabulary declared on it. If the id does not exist, memchat stops at startup
and says the workspace is either missing or not reachable with your credential,
because the platform answers both cases the same way.

At startup memchat shows the vocabulary in effect for the workspace. Reading it
needs the `agent.vocabulary:read` permission. If your credential lacks it,
memchat says so and starts anyway.

## Tests

```sh
pip install ".[test]"
pytest
```

The tests run against an in-process fake Jennah reached through the SDK's own
client, so they need no network or credentials.

## License

Apache-2.0
