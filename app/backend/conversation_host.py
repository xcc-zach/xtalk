"""Application-owned login, history and upload endpoints."""

from pathlib import Path
from typing import Any
import uuid
import json
import asyncio
from fastapi import WebSocket, Request, HTTPException, File, Form, UploadFile, status
from .user_auth import JWTAuth, extract_bearer_token, resolve_auth_config
from .conversation_store import PersistenceStore
from .knowledge_store import KnowledgeStore
from .recording_store import RecordingStore


class ApplicationHost:
    """Compose application identity/storage with a pure conversation runtime.

    Parameters
    ----------
    runtime : Xtalk
        Core conversation runtime.
    data_dir : str
        Host-owned data directory.
    """

    def __init__(
        self,
        runtime,
        data_dir="data",
        *,
        user_id=None,
        embeddings=None,
        record_audio=False,
    ):
        if user_id is not None and (
            not isinstance(user_id, str) or not user_id.strip()
        ):
            raise ValueError("user_id must be a non-empty string")
        if user_id is not None:
            user_id = user_id.strip()
        self.runtime = runtime
        self._data_dir = Path(data_dir)
        self.knowledge = (
            KnowledgeStore(self._data_dir / "sessions", embeddings)
            if embeddings is not None
            else None
        )
        self._record_audio = record_audio
        self._connections = {}
        self._runtime_sessions = {}
        self._user_id = user_id
        self._persistence = PersistenceStore(Path(data_dir) / "chat_history.sqlite3")
        secret, ttl = resolve_auth_config({})
        self._auth = JWTAuth(secret=secret, ttl_seconds=ttl)

    async def embed_text(self, *, session_id: str, text: str, user_id: str):
        """Index uploaded text after verifying application ownership.

        Parameters
        ----------
        session_id : str
            Application conversation identifier.
        text : str
            Uploaded text document.
        user_id : str
            Authenticated owner.
        """
        if not self._persistence.user_owns_session(user_id, session_id):
            raise ValueError("Session not found")
        if self.knowledge is None:
            raise ValueError("Configure an embedding model on the application host")
        return await self.knowledge.add(conversation_id=session_id, text=text)

    def runtime_session_id(self, conversation_id: str) -> str:
        """Resolve an application conversation to its active runtime identifier.

        Parameters
        ----------
        conversation_id : str
            Application conversation identifier.
        """
        return self._runtime_sessions.get(conversation_id, conversation_id)

    async def delete_conversation(self, *, conversation_id: str) -> None:
        """Stop active connections before deleting an owned conversation.

        Parameters
        ----------
        conversation_id : str
            Application conversation identifier to delete.
        """
        if not self._persistence.user_owns_session(self._user_id, conversation_id):
            raise KeyError(conversation_id)
        tasks = tuple(self._connections.get(conversation_id, ()))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with self._persistence._lock, self._persistence._connect() as connection:
            connection.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                (conversation_id, self._user_id),
            )

    def _login(self) -> dict[str, Any]:
        user_id = self._user_id or str(uuid.uuid4())
        user = (
            self._persistence.ensure_user(user_id)
            if self._persistence is not None
            else {"id": user_id}
        )
        return {
            "access_token": self._auth.issue_token(sub=user_id),
            "user": user,
        }

    def _verify_access_token(self, token: str) -> str:
        user_id = self._auth.verify_token(token)
        if self._persistence is not None:
            self._persistence.ensure_user(user_id)
        return user_id

    def _list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        if self._persistence is None:
            return []
        self._persistence.ensure_user(user_id)
        return self._persistence.list_sessions(user_id)

    def _get_session_detail(self, user_id: str, session_id: str) -> dict[str, Any]:
        if self._persistence is None:
            raise KeyError(session_id)
        self._persistence.ensure_user(user_id)
        return self._persistence.get_session_detail(user_id, session_id)

    def mount_routes(
        self,
        app: Any,
        *,
        login_path: str = "/api/auth/login",
        sessions_path: str = "/api/sessions",
        session_detail_path: str = "/api/sessions/{session_id}",
        upload_path: str = "/api/upload",
        ws_path: str = "/ws",
    ) -> None:
        """Mount the built-in auth, session, upload, and websocket routes."""

        def _require_http_user(request: Request) -> str:
            token = extract_bearer_token(request.headers.get("authorization"))
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing bearer token",
                )
            try:
                return self._verify_access_token(token)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(exc),
                ) from exc

        def _require_ws_user(websocket: WebSocket) -> str:
            token = websocket.query_params.get("access_token")
            if token is None:
                token = extract_bearer_token(websocket.headers.get("authorization"))
            if not token:
                raise ValueError("Missing access token")
            return self._verify_access_token(token)

        @app.post(login_path)
        async def _login_route() -> dict[str, Any]:
            return self._login()

        @app.get(sessions_path)
        async def _list_sessions_route(request: Request) -> dict[str, Any]:
            user_id = _require_http_user(request)
            return {"sessions": self._list_sessions(user_id)}

        @app.post(sessions_path)
        async def _create_session_route(request: Request) -> dict[str, Any]:
            user_id = _require_http_user(request)
            return self._persistence.create_session(user_id)

        @app.get(session_detail_path)
        async def _session_detail_route(
            request: Request, session_id: str
        ) -> dict[str, Any]:
            user_id = _require_http_user(request)
            try:
                return self._get_session_detail(user_id, session_id)
            except KeyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
                ) from exc

        @app.post(upload_path)
        async def _upload_route(
            request: Request,
            session_id: str = Form(...),
            file: UploadFile = File(...),
        ) -> dict[str, str]:
            user_id = _require_http_user(request)
            content_type = (file.content_type or "").lower()
            is_text = content_type.startswith("text/") if content_type else False
            if content_type and not is_text:
                raise HTTPException(
                    status_code=400, detail="Only text files are supported."
                )
            text = (await file.read()).decode("utf-8", errors="ignore")
            try:
                await self.embed_text(session_id=session_id, text=text, user_id=user_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"status": "ok"}

        realtime = self.runtime.create_asgi_app()

        @app.websocket("/v1/realtime")
        async def _realtime_route(websocket: WebSocket) -> None:
            try:
                user_id = _require_ws_user(websocket)
            except ValueError:
                await websocket.close(code=1008)
                return
            conversation_id = websocket.query_params.get("conversation_id")
            if conversation_id:
                if not self._persistence.user_owns_session(user_id, conversation_id):
                    await websocket.close(code=1008)
                    return
            else:
                conversation_id = self._persistence.create_session(user_id)[
                    "session_id"
                ]

            recording = (
                RecordingStore(
                    self._data_dir
                    / "sessions"
                    / conversation_id
                    / ("audio_" + uuid.uuid4().hex + ".wav")
                )
                if self._record_audio
                else None
            )

            pending_text = []

            async def receive():
                message = await websocket.receive()
                if message.get("text"):
                    try:
                        event = json.loads(message["text"])
                    except json.JSONDecodeError:
                        return message
                    if (
                        isinstance(event, dict)
                        and event.get("type") == "conversation.item.create"
                    ):
                        extension = event.get("xtalk", {})
                        item = event.get("item", {})
                        if (
                            isinstance(extension, dict)
                            and extension.get("generate_response") is True
                            and isinstance(item, dict)
                            and item.get("role") == "user"
                        ):
                            pending_text.append(item)
                return message

            async def send(message):
                if message.get("text"):
                    event = json.loads(message["text"])
                    if (
                        event["type"] == "session.audio.mix.delta"
                        and recording is not None
                    ):
                        recording.append(event=event)
                    if event["type"] == "session.created":
                        self._runtime_sessions[conversation_id] = event["session"]["id"]
                        event["session"]["application_conversation_id"] = (
                            conversation_id
                        )
                        message = dict(message, text=json.dumps(event))
                    if (
                        event["type"] == "conversation.item.done"
                        and event["item"].get("role") == "user"
                    ):
                        completed = event["item"]
                        matched = next(
                            (
                                item
                                for item in pending_text
                                if item.get("content") == completed.get("content")
                                and (
                                    not item.get("id") or item["id"] == completed["id"]
                                )
                            ),
                            None,
                        )
                        if matched is not None:
                            pending_text.remove(matched)
                            text = "".join(
                                part["text"] for part in completed["content"]
                            )
                            self._persistence.append_message(
                                user_id=user_id,
                                session_id=conversation_id,
                                role="user",
                                content=text,
                            )
                    if (
                        event["type"]
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        self._persistence.append_message(
                            user_id=user_id,
                            session_id=conversation_id,
                            role="user",
                            content=event["transcript"],
                        )
                    elif event["type"] in {
                        "response.output_audio.playback.updated",
                        "response.output_audio.playback.done",
                        "response.output_text.done",
                    }:
                        self._persistence.upsert_assistant(
                            user_id=user_id,
                            session_id=conversation_id,
                            item_id=event.get("item_id")
                            or "item_" + event["response_id"],
                            content=event.get("text", ""),
                        )
                await websocket.send(message)

            task = asyncio.current_task()
            connections = self._connections.setdefault(conversation_id, set())
            connections.add(task)
            try:
                await realtime(websocket.scope, receive, send)
            finally:
                connections.discard(task)
                if not connections:
                    self._connections.pop(conversation_id, None)
                if recording is not None:
                    recording.close()
