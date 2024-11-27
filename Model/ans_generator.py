from typing import Any, Dict, List, Optional

from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from Model.prompts import PROMPTS


def format_docs(docs: List[Document]):
    return "\n".join(doc.page_content for doc in docs)


class ChatLlamaCppManager:
    """
    Singleton 實現，用於管理 ChatLlamaCpp 的單例實例。
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ChatLlamaCppManager, cls).__new__(cls)
            cls._instance._initialize()  # 初始化內部狀態
        return cls._instance

    def _initialize(self):
        """
        初始化內部變量。
        """
        self._llm_instance = None
        self._current_params = None

    def get_instance(self, llm_kwargs: Optional[Dict[str, Any]] = None) -> ChatLlamaCpp:
        """
        獲取或更新 ChatLlamaCpp 實例。

        Args:
            llm_kwargs (Optional[Dict[str, Any]]): 覆蓋默認參數的字典。

        Returns:
            ChatLlamaCpp: ChatLlamaCpp 的實例。
        """
        default_params = {
            "temperature": 0.01,
            "top_p": 0.95,
            "model_path": "breeze-7b-instruct-v1_0-q8_0.gguf",
            "n_ctx": 4096,
            "max_token": 400,
            "n_gpu_layers": -1,
            "n_batch": 128,
            "verbose": False,
        }

        # 合併默認參數和用戶傳入參數
        if llm_kwargs:
            merged_params = default_params.copy()
            merged_params.update(llm_kwargs)
        else:
            merged_params = default_params

        # 若實例不存在或參數變更，重新初始化
        if self._llm_instance is None or self._current_params != merged_params:
            self._llm_instance = ChatLlamaCpp(**merged_params)
            self._current_params = merged_params

        return self._llm_instance


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
    llm_manager = ChatLlamaCppManager()
    llm = llm_manager.get_instance(llm_kwargs)

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
