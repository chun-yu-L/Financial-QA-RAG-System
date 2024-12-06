from typing import Any, Dict, List, Union

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from typing_extensions import TypedDict


class QAState(TypedDict):
    question: Dict[str, Any]
    doc_set: Dict[str, Any]
    client: QdrantClient
    retrieve_doc: Union[str, List[Document]]
    answer: Dict[str, Any]
