from typing import Any, Dict, Optional

from langchain_community.chat_models import ChatLlamaCpp


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
            "n_batch": 512,
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
