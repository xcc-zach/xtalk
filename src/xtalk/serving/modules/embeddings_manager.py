import asyncio
import os
import shutil
from typing import Any

from ...log_utils import logger
from ..event_bus import EventBus
from ..interfaces import Manager
from ...models import Embeddings, Models
from ..events import (
    EmbeddingStatusUpdated,
    TextForEmbeddingReady,
)


class EmbeddingsManager(Manager):
    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        self.models = models
        # Session-level config shared with managers
        self.config: dict[str, Any] = config or {}

    @Manager.event_handler(TextForEmbeddingReady, priority=70)
    async def _on_text_for_embedding_ready(self, event: TextForEmbeddingReady) -> None:
        """Handle text that needs to be embedded."""
        text = event.text or ""
        if not text.strip():
            return
        await self.event_bus.publish(
            EmbeddingStatusUpdated(
                session_id=self.session_id,
                status="processing",
                text=text,
            ),
            wait_for_completion=True,
        )

        db = await self._run_embedding_job(text)
        await self.event_bus.publish(
            EmbeddingStatusUpdated(
                session_id=self.session_id,
                status="finished",
                text=text,
                vector_store_instance=db,
            ),
            wait_for_completion=True,
        )

    def _resolve_data_dir(self) -> str | None:
        """Resolve data_dir from config."""
        data_dir = self.config.get("data_dir")
        if not data_dir:
            return None
        return str(data_dir)

    def _resolve_session_root(self) -> str | None:
        """Return session root directory under data_dir."""
        data_dir = self._resolve_data_dir()
        if not data_dir:
            return None
        return os.path.join(data_dir, "sessions", self.session_id)

    def _resolve_persist_directory(self) -> str | None:
        """Return persistence directory for embeddings."""
        session_root = self._resolve_session_root()
        if not session_root:
            return None
        return os.path.join(session_root, "embeddings")

    async def _run_embedding_job(self, text: str) -> Any | None:
        """Write text to session-level Chroma vector store."""
        # Fetch embeddings model
        embeddings_model = self.models.get(Embeddings)
        if embeddings_model is None:
            logger.warning(
                "Embeddings model is not configured on models, skip embedding."
            )
            return None

        persist_directory = self._resolve_persist_directory()
        if not persist_directory:
            logger.warning("No data_dir configured, skip embedding.")
            return None

        os.makedirs(persist_directory, exist_ok=True)

        try:
            from langchain_chroma import Chroma  # type: ignore
            from langchain_text_splitters import (  # type: ignore
                RecursiveCharacterTextSplitter,
            )
        except Exception as e:
            logger.error(
                "Embedding backend unavailable: missing langchain_chroma/text_splitters: %s",
                e,
            )
            return None

        # Use same splitting strategy as retrieval tool
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=32,
            separators=[
                "\n\n",
                "。",
                "！",
                "？",
                "；",
                "\n",
                " ",
                "",
                ".",
                "!",
                "?",
                ";",
            ],
            keep_separator=True,
            strip_whitespace=True,
        )

        docs = text_splitter.create_documents(
            [text],
            metadatas=[{"session_id": self.session_id}],
        )

        # Always open/create Chroma with given persist_directory
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings_model,
        )

        # Prefer async API; otherwise run sync method in thread pool
        if hasattr(db, "aadd_documents"):
            try:
                await db.aadd_documents(docs, embedding=embeddings_model)  # type: ignore[arg-type]
            except TypeError:
                await db.aadd_documents(docs)  # type: ignore[func-returns-value]
        else:
            await asyncio.to_thread(db.add_documents, docs)

        return db

    async def shutdown(self) -> None:
        """Remove per-session embedding directory on shutdown."""
        session_root = self._resolve_session_root()
        if not session_root:
            return
        # Only delete this session directory to avoid removing other data
        if os.path.isdir(session_root):
            shutil.rmtree(session_root, ignore_errors=False)
