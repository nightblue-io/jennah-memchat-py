import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fake import API_KEY, FakeJennah, Server  # noqa: E402

import jennah  # noqa: E402


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A machine with no Jennah credential: no env key, and an empty config dir
    where the ``jnh login`` session would be."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("JENNAH_API_KEY", raising=False)
    return tmp_path


@pytest.fixture
def fake():
    f = FakeJennah()
    srv = Server(f)
    f.endpoint = srv.endpoint
    yield f
    srv.stop()


@pytest.fixture
def client(fake):
    c = jennah.Client(endpoint=fake.endpoint, insecure=True, api_key=API_KEY)
    yield c
    c.close()
