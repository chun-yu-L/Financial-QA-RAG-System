from typing import List

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from Model.prompts import PROMPTS


def format_docs(docs: List[Document]):
    return "\n".join(doc.page_content for doc in docs)


def answer_generation(question, retrieved_docs: List[Document], llm=None):
    if llm is None:
        llm = ChatLlamaCpp(
            temperature=0.01,
            top_p=0.95,
            model_path="breeze-7b-instruct-v1_0-q8_0.gguf",
            n_ctx=2048,
            max_token=400,
            n_gpu_layers=-1,
            n_batch=512,
        )

    prompt_mapping = {"faq": PROMPTS["faq_ans"], "insurance": PROMPTS["insurance_ans"]}

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
