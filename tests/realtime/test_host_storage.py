"""Preserve host-owned storage while core conversations stay temporary."""

import asyncio
import base64
import wave

from xtalk.models import Embeddings
from examples.sample_app.host.knowledge_store import KnowledgeStore
from examples.sample_app.host.recording_store import RecordingStore


class LocalEmbeddings(Embeddings):
    """Deterministic vectors for a small offline retrieval acceptance fixture."""

    def embed_documents(self, texts):
        """Embed the fixture's topic words without a network provider."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        """Place the two topic words on orthogonal vector axes."""
        return [float("apple" in text), float("ocean" in text), 0.1]


def test_host_knowledge_upload_remains_searchable_after_reopening(tmp_path):
    async def run():
        knowledge = KnowledgeStore(tmp_path, LocalEmbeddings())
        await knowledge.add(conversation_id="owned", text="red apple")
        await knowledge.add(conversation_id="owned", text="blue ocean")
        reopened = KnowledgeStore(tmp_path, LocalEmbeddings()).open(
            conversation_id="owned"
        )
        assert reopened.similarity_search("apple", k=1)[0].page_content == "red apple"

    asyncio.run(run())


def test_host_recording_preserves_mixed_pcm_and_wave_metadata(tmp_path):
    path = tmp_path / "recording.wav"
    recorder = RecordingStore(path)
    pcm = bytes(range(64))
    recorder.append(
        event={
            "delta": base64.b64encode(pcm).decode(),
            "sample_rate": 48000,
            "channels": 2,
        }
    )
    recorder.close()
    with wave.open(str(path)) as audio:
        assert audio.getframerate() == 48000
        assert audio.getnchannels() == 2
        assert audio.readframes(audio.getnframes()) == pcm


def test_host_applies_playback_snapshots_without_storing_generated_audio_text(tmp_path):
    """Repeated acknowledgements retain one complete heard-text snapshot."""
    from examples.sample_app.host.persistence import PersistenceStore

    store = PersistenceStore(tmp_path / "history.sqlite3")
    conversation = store.create_session("owner")["session_id"]
    store.append_message(
        user_id="owner", session_id=conversation, role="user", content="question"
    )
    for text in ["heard", "heard prefix", "heard prefix"]:
        store.upsert_assistant(
            user_id="owner",
            session_id=conversation,
            item_id="item_response",
            content=text,
        )
    reopened = PersistenceStore(tmp_path / "history.sqlite3")
    assert reopened.list_messages("owner", conversation) == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "heard prefix"},
    ]


def test_host_authentication_history_and_reconnection(tmp_path):
    """Keep business identity stable across temporary authenticated connections."""
    from contextlib import asynccontextmanager
    from asgi_lifespan import LifespanManager
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    import pytest
    from unittest.mock import Mock
    from xtalk import Models, Service, Xtalk
    from xtalk.models import Agent
    from examples.sample_app.host.application import ApplicationHost

    from xtalk.serving.modules.llm_agent_context_manager import LLMAgentContextManager

    runtime = Xtalk(service_prototype=Service(
        models=Models({Agent: Mock(spec=Agent)}),
        manager_classes=[LLMAgentContextManager],
    ))
    realtime = runtime.create_asgi_app()

    @asynccontextmanager
    async def lifespan(app):
        async with LifespanManager(realtime):
            yield

    app = FastAPI(lifespan=lifespan)
    host = ApplicationHost(runtime, data_dir=tmp_path)
    host.mount_routes(app)
    with TestClient(app) as client:
        assert client.get("/api/sessions").status_code == 401
        login = client.post("/api/auth/login").json()
        headers = {"Authorization": "Bearer " + login["access_token"]}
        conversation = client.post("/api/sessions", headers=headers).json()[
            "session_id"
        ]
        url = (
            "/v1/realtime?conversation_id="
            + conversation
            + "&access_token="
            + login["access_token"]
        )
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/v1/realtime?conversation_id=" + conversation
            ):
                pass
        ids = []
        for index in range(2):
            with client.websocket_connect(url) as socket:
                created = socket.receive_json()
                ids.append(created["session"]["id"])
                assert created["session"]["application_conversation_id"] == conversation
                socket.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "id": "history-" + str(index),
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "restored history"}
                            ],
                        },
                    }
                )
                assert socket.receive_json()["type"] == "conversation.item.added"
                assert socket.receive_json()["type"] == "conversation.item.done"
                socket.close()

                async def disconnected():
                    while host._connections.get(conversation):
                        await asyncio.sleep(0)

                client.portal.call(asyncio.wait_for, disconnected(), 2)
        assert ids[0] != ids[1]
        # Context replay is accepted but never duplicated in business history.
        detail = client.get("/api/sessions/" + conversation, headers=headers).json()
        assert detail["messages"] == []
        other = client.post("/api/auth/login").json()["access_token"]
        assert (
            client.get(
                "/api/sessions/" + conversation,
                headers={"Authorization": "Bearer " + other},
            ).status_code
            == 404
        )
