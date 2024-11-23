from typing import List

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from Model.prompts import PROMPTS


def format_docs(docs: List[Document]):
    return "\n".join(doc.page_content for doc in docs)


def answer_generation(question, retrieved_docs: List[Document], llm_kwargs=None) -> str:
    """
    根據問題和檢索的文件生成答案。

    Args:
        question: 問題字典，應包括 'category' 和 'query' 鍵。
        retrieved_docs (List[Document]): 檢索到的文檔列表。
        llm_kwargs (Optional[Dict[str, Any]]): `ChatLlamaCpp` 的參數字典，用於覆蓋默認設置。

    Returns:
        str: 生成的答案。
    """
    default_llm_params = {
        "temperature": 0.01,
        "top_p": 0.95,
        "model_path": "breeze-7b-instruct-v1_0-q8_0.gguf",
        "n_ctx": 4096,
        "max_token": 400,
        "n_gpu_layers": -1,
        "n_batch": 128,
        "verbose": False,
    }

    if llm_kwargs:
        default_llm_params.update(llm_kwargs)

    llm = ChatLlamaCpp(**default_llm_params)

    prompt_mapping = {
        "faq": PROMPTS["faq_ans"],
        "finance": PROMPTS["finance_ans"],
        "insurance": PROMPTS["insurance_ans"],
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
