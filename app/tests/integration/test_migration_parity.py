"""Exercise desktop-owned identity/history against the native Realtime runtime."""

from contextlib import asynccontextmanager
from unittest.mock import Mock
import asyncio

from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from fastapi.testclient import TestClient
from xtalk import Xtalk, Models, Service
from xtalk.models import Agent
from xtalk.serving.modules.llm_agent_context_manager import LLMAgentContextManager
from backend.conversation_host import ApplicationHost


def test_desktop_history_restores_into_temporary_realtime_context(tmp_path):
    """Maintain anonymous ownership, persisted heard text and accepted replay."""
    runtime = Xtalk(
        service_prototype=Service(
            models=Models({Agent: Mock(spec=Agent)}),
            manager_classes=[LLMAgentContextManager],
        )
    )
    realtime = runtime.create_asgi_app()

    @asynccontextmanager
    async def lifespan(app):
        async with LifespanManager(realtime):
            yield

    app = FastAPI(lifespan=lifespan)
    host = ApplicationHost(runtime, data_dir=tmp_path, user_id="desktop-owner")
    host.mount_routes(app)
    with TestClient(app) as client:
        first = client.post("/api/auth/login").json()
        second = client.post("/api/auth/login").json()
        assert first["user"]["id"] == second["user"]["id"] == "desktop-owner"
        headers = {"Authorization": "Bearer " + first["access_token"]}
        conversation = client.post("/api/sessions", headers=headers).json()[
            "session_id"
        ]
        host._persistence.append_message(
            user_id="desktop-owner",
            session_id=conversation,
            role="user",
            content="question",
        )
        host._persistence.upsert_assistant(
            user_id="desktop-owner",
            session_id=conversation,
            item_id="heard",
            content="heard prefix",
        )
        detail = client.get("/api/sessions/" + conversation, headers=headers).json()
        assert detail["messages"] == [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "heard prefix"},
        ]
        with client.websocket_connect(
            "/v1/realtime?conversation_id="
            + conversation
            + "&access_token="
            + second["access_token"]
        ) as socket:
            created = socket.receive_json()
            assert created["session"]["application_conversation_id"] == conversation
            assert created["session"]["id"] != conversation
            for message in detail["messages"]:
                socket.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": message["role"],
                            "content": [
                                {
                                    "type": "text"
                                    if message["role"] == "assistant"
                                    else "input_text",
                                    "text": message["content"],
                                }
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
        assert (
            client.get("/api/sessions/" + conversation, headers=headers).json()[
                "messages"
            ]
            == detail["messages"]
        )
