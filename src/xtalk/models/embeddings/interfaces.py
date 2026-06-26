from langchain_core.embeddings import Embeddings as _LangChainEmbeddings

from ..registry import model_type


@model_type(aliases=["embeddings"])
class Embeddings(_LangChainEmbeddings):
    """Interface marker for embedding models."""

__all__ = ["Embeddings"]
