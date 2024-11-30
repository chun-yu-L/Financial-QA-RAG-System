from typing import Any, Dict

from qdrant_client import QdrantClient
from typing_extensions import TypedDict


class QAState(TypedDict):
    question: Dict[str, Any]
    doc_set: Dict[str, Any]
    client: QdrantClient
    answer: Dict[str, Any]
