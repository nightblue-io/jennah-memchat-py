"""An in-process Jennah for tests: the agent and memory services over real gRPC.

Tests reach it through the SDK's own ``Client(endpoint=..., insecure=True,
api_key=...)``, so every call goes through the same stubs, interceptors and
deadlines a live run does. The fake records each request with the deadline the
client attached, and each method's answer can be set, delayed, or held open.
"""

from __future__ import annotations

import threading
import time
from concurrent import futures
from typing import Callable, Optional

import grpc

from jennah.agent.v1 import agent_pb2, agent_pb2_grpc, memory_pb2, memory_pb2_grpc

API_KEY = "jennah_sk_test_fake"


class Call:
    def __init__(self, method: str, request, deadline: Optional[float], started: float) -> None:
        self.method = method
        self.request = request
        self.deadline = deadline  # seconds remaining when the call arrived, or None
        self.started = started
        self.ended: Optional[float] = None


class FakeJennah(agent_pb2_grpc.AgentServiceServicer, memory_pb2_grpc.MemoryServiceServicer):
    def __init__(self) -> None:
        self.mu = threading.Lock()
        self.calls: list[Call] = []
        self.agents: set[str] = set()
        self.query = memory_pb2.QueryMemoryResponse()
        # One response per InspectMemory call, in order; the last one repeats.
        self.inspect_pages: list[memory_pb2.InspectMemoryResponse] = [memory_pb2.InspectMemoryResponse()]
        self.vocabulary: memory_pb2.GetMemoryVocabularyResponse | grpc.StatusCode = (
            memory_pb2.GetMemoryVocabularyResponse()
        )
        self.commit = memory_pb2.CommitMemoryResponse()
        # form(request) -> response. Replay by formation key is the fake's job,
        # the way it is the platform's: a repeated key returns the first receipt.
        self.form: Callable[[memory_pb2.FormMemoryRequest], memory_pb2.FormMemoryResponse] = (
            lambda req: memory_pb2.FormMemoryResponse()
        )
        self.formed: dict[str, memory_pb2.FormMemoryResponse] = {}
        self.extractions = 0
        self.delay: dict[str, float] = {}
        self.hold: dict[str, threading.Event] = {}
        self.fail: dict[str, grpc.StatusCode] = {}

    # --- plumbing ---

    def _enter(self, method: str, request, context) -> Call:
        remaining = context.time_remaining()
        call = Call(method, request, remaining, time.monotonic())
        with self.mu:
            self.calls.append(call)
        auth = dict(context.invocation_metadata()).get("authorization", "")
        if auth != "Bearer " + API_KEY:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "the credential is invalid")
        if method in self.delay:
            time.sleep(self.delay[method])
        if method in self.hold:
            self.hold[method].wait(10)
        if method in self.fail:
            context.abort(self.fail[method], f"{method} failed in the fake")
        return call

    def _leave(self, call: Call, resp):
        call.ended = time.monotonic()
        return resp

    def of(self, method: str) -> list[Call]:
        with self.mu:
            return [c for c in self.calls if c.method == method]

    # --- AgentService ---

    def GetAgent(self, request, context):
        call = self._enter("GetAgent", request, context)
        if request.agent_instance_id not in self.agents:
            context.abort(grpc.StatusCode.NOT_FOUND, "agent not found")
        return self._leave(call, agent_pb2.GetAgentResponse(
            agent=agent_pb2.AgentInstance(agent_instance_id=request.agent_instance_id)))

    def CreateAgent(self, request, context):
        call = self._enter("CreateAgent", request, context)
        self.agents.add(request.agent_instance_id)
        return self._leave(call, agent_pb2.CreateAgentResponse(
            agent=agent_pb2.AgentInstance(agent_instance_id=request.agent_instance_id)))

    # --- MemoryService ---

    def QueryMemory(self, request, context):
        call = self._enter("QueryMemory", request, context)
        return self._leave(call, self.query)

    def InspectMemory(self, request, context):
        call = self._enter("InspectMemory", request, context)
        n = len(self.of("InspectMemory")) - 1
        return self._leave(call, self.inspect_pages[min(n, len(self.inspect_pages) - 1)])

    def GetMemoryVocabulary(self, request, context):
        call = self._enter("GetMemoryVocabulary", request, context)
        if isinstance(self.vocabulary, grpc.StatusCode):
            context.abort(self.vocabulary, "vocabulary read refused in the fake")
        return self._leave(call, self.vocabulary)

    def CommitMemory(self, request, context):
        call = self._enter("CommitMemory", request, context)
        return self._leave(call, self.commit)

    def FormMemory(self, request, context):
        call = self._enter("FormMemory", request, context)
        with self.mu:
            prior = self.formed.get(request.formation_key) if request.formation_key else None
        if prior is not None:
            return self._leave(call, prior)
        resp = self.form(request)
        with self.mu:
            self.extractions += 1
            if request.formation_key:
                self.formed[request.formation_key] = resp
        return self._leave(call, resp)


class Server:
    """A running fake. ``endpoint`` is what a Client is pointed at."""

    def __init__(self, fake: FakeJennah) -> None:
        self.fake = fake
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        agent_pb2_grpc.add_AgentServiceServicer_to_server(fake, self._server)
        memory_pb2_grpc.add_MemoryServiceServicer_to_server(fake, self._server)
        port = self._server.add_insecure_port("127.0.0.1:0")
        self.endpoint = f"127.0.0.1:{port}"
        self._server.start()

    def stop(self) -> None:
        for ev in self.fake.hold.values():
            ev.set()
        self._server.stop(None)
