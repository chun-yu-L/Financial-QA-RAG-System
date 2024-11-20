from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from Model.prompts import PROMPTS


def format_docs(docs: List[Document]):
    return "\n".join(doc.page_content for doc in docs)


def get_qa_answer(question, llm, retrieved_docs: List[Document]):
    prompt_mapping = {
        "faq": PROMPTS["faq_ans"],
    }

    prompt = prompt_mapping.get(question["category"])

    if not prompt:
        raise ValueError(f"Unsupported category: {question['category']}")

    chain = (
        {
            "context": lambda x: format_docs(retrieved_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question["query"])
