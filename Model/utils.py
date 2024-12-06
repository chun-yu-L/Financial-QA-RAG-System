from typing import List

import tiktoken
from langchain_core.documents import Document


def count_tokens(text: str) -> int:
    """Returns the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))


def format_docs(docs: List[Document]):
    return "\n".join(doc.page_content for doc in docs)
