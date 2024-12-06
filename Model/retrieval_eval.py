from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from Model.llm import ChatLlamaCppManager
from Model.utils import format_docs

PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
            You are a grader assessing the relevance of a retrieved document to a user question. 

            **Instructions:**
            - Analyze the document and the user question carefully, including any Markdown tables or other structured data in the document.
            - If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
            - Respond strictly with a binary score: 'Yes' or 'No'.
            - Do not provide explanations, additional context, or any other output beyond 'Yes' or 'No'.
            """,
        ),
        (
            "human",
            """
            Please evaluate the following:

            **Query:** {query}

            <document>
            {document}
            </document>

            **Task:** Determine if the document contains keyword(s) or semantic meaning that makes it relevant to the user question. 
            Respond with 'Yes' if relevant, otherwise respond with 'No'.
            """,
        ),
    ]
)


class DocumentContainAnswer(BaseModel):
    """
    判斷文檔是否包含問題的答案的回應。

    Attributes:
        answer (str): 指示文檔是否包含問題的解答，值必須為 'Yes' 或 'No'。
    """

    answer: str = Field(
        ...,
        description="Indicates whether the document contains the answer to the question. Must be 'Yes' or 'No'.",
    )


def document_contains_answer_check(
    question, retrieved_docs: List[Document], llm_kwargs=None
):
    """
    使用 LLM 判斷當前文檔是否包含解答。

    Args:
        question: 問題字典，應包括 'category' 和 'query' 鍵。
        retrieved_docs (List[Document]): 檢索到的文檔列表。
        llm_kwargs (Optional[Dict[str, Any]]): `ChatLlamaCpp` 的參數字典，用於覆蓋默認設置。

    Returns:
        str: 回應 'Yes' 表示文檔包含解答，'No' 表示不包含。
    """
    llm_manager = ChatLlamaCppManager()
    llm = llm_manager.get_instance(llm_kwargs)
    structured_llm = llm.with_structured_output(DocumentContainAnswer)

    # 如果 retrieved_docs 是字符串，轉換為單元素列表
    if isinstance(retrieved_docs, str):
        retrieved_docs = [Document(page_content=retrieved_docs)]

    chain = PROMPT | structured_llm
    return chain.invoke(
        {"query": question["query"], "document": format_docs(retrieved_docs)}
    ).answer
