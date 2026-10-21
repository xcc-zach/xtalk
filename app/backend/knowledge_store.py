"""Host-owned document splitting and persistent knowledge indexing."""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeStore:
    """Manage knowledge resources outside the conversation SDK.

    Parameters
    ----------
    root : pathlib.Path
        Application-owned directory for conversation knowledge.
    embeddings : Embeddings
        Explicitly configured embedding model.
    """

    def __init__(self, root: Path, embeddings):
        self.root = root
        self.embeddings = embeddings

    def open(self, *, conversation_id: str):
        """Open a conversation's vector store for upload or retrieval.

        Parameters
        ----------
        conversation_id : str
            Validated application conversation identifier.
        """
        directory = self.root / conversation_id / "embeddings"
        directory.mkdir(parents=True, exist_ok=True)
        return Chroma(
            persist_directory=str(directory), embedding_function=self.embeddings
        )

    async def add(self, *, conversation_id: str, text: str):
        """Split and index an uploaded document in its host-owned store.

        Parameters
        ----------
        conversation_id : str
            Validated application conversation identifier.
        text : str
            Uploaded document content.
        """
        splitter = RecursiveCharacterTextSplitter(
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
        )
        documents = splitter.create_documents(
            [text], metadatas=[{"session_id": conversation_id}]
        )
        store = self.open(conversation_id=conversation_id)
        if documents:
            await store.aadd_documents(documents)
        return store
