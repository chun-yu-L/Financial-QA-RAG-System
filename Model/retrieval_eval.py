import json
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from Model.llm import ChatLlamaCppManager

PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
            You are a highly accurate and reliable assistant. Your task is to determine whether a given document contains 
            information that directly answers a specific query. 

            **Instructions:**
            - Carefully analyze both the query and the document provided.
            - Respond only with a single word: 'Yes' or 'No'.
            - Do not provide explanations, additional context, or any other output beyond 'Yes' or 'No'.
            """,
        ),
        (
            "human",
            """
            Please evaluate the following:

            **Query:** {query}
            **Document:** {document}

            **Task:** Determine if the document contains information that directly answers the query. 
            Respond with 'Yes' if the document contains the answer to the query. Otherwise, respond with 'No'.
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

    chain = PROMPT | structured_llm
    return chain.invoke(
        {"query": question["query"], "document": retrieved_docs[0].page_content}
    ).answer


class RAGRelevanceChecker:
    def __init__(
        self, azure_deployment_name, azure_endpoint, azure_key, azure_deployment
    ):
        """
        Initialize Azure OpenAI client for relevance checking

        :param azure_endpoint: Azure OpenAI service endpoint
        :param azure_key: API key for authentication
        :param azure_deployment: Deployment name for GPT-4
        """
        self.client = AzureOpenAI(
            azure_deployment_name=azure_deployment_name,
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version="2024-02-15-preview",
        )
        self.deployment = azure_deployment

    def check_document_relevance(self, query: str, idx: int, document: str) -> bool:
        """
        Use Azure OpenAI to check document relevance

        :param query: User's original question
        :param idx: Document index
        :param document: Retrieved document text
        :return: Boolean indicating document relevance
        """
        relevance_prompt = f"""
        Determine if the following document contains information directly relevant to answering the query. 
        
        Query: {query}
        Document: {document}
        
        Respond ONLY with a JSON object containing a 'relevance' key:
        - If the document is highly relevant, return: {{"relevance": "Yes"}}
        - If the document is not relevant, return: {{"relevance": "No"}}
        """

        try:
            # Generate response using Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": relevance_prompt}],
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            llm_response = response.choices[0].message.content
            relevance_data = json.loads(llm_response)
            print(idx, relevance_data)
            return relevance_data.get("relevance", "No") == "Yes"

        except Exception as e:
            print(f"{idx} error: {e}")
            return False

    def eval_response(self, query: str, retrieved_docs: list) -> str:
        """
        Evaluate response using RAG with relevance checking

        :param query: User's original question
        :param retrieved_docs: List of retrieved documents
        :return: Result of relevance checking
        """
        # Iterate through top documents
        for i, doc in enumerate(retrieved_docs[:3], 1):
            # Check document relevance
            if self.check_document_relevance(query, i, doc):
                return "Passed"

        # Fallback if no relevant document found
        return "不知道"


if __name__ == "__main__":
    load_dotenv()

    relevance_checker = RAGRelevanceChecker(
        azure_deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    # relevance_checker.eval_response(
    #     query="智邦科技股份有限公司2023年第1季的綜合損益總額是多少？",
    #     retrieved_docs=[]
    # )
