from jennah.agent.v1 import agent_pb2, memory_pb2


def test_client_reaches_the_fake(fake, client):
    fake.agents.add("demo.x")
    resp = client.agents.GetAgent(agent_pb2.GetAgentRequest(agent_instance_id="demo.x"), timeout=5)
    assert resp.agent.agent_instance_id == "demo.x"
    client.memory.FormMemory(memory_pb2.FormMemoryRequest(scope_id="demo.x", formation_key="k"), timeout=7)
    got = fake.of("FormMemory")[0]
    assert got.request.formation_key == "k"
    assert 5 < got.deadline <= 8  # gRPC rounds the wire deadline up
