from typing import Any, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from typing_extensions import TypedDict


class QAState(TypedDict):
    question: Dict[str, Any]
    doc_set: Dict[str, Any]
    embedding_model: HuggingFaceEmbeddings
    client: QdrantClient
    answer: Dict[str, Any]
