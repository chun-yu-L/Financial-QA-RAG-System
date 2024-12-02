import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

class RAGRelevanceChecker:
    def __init__(self, azure_deployment_name, azure_endpoint, azure_key, azure_deployment):
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
            api_version="2024-02-15-preview"
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
        Determine if the following document contains information 
        directly relevant to answering the query. 
        
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
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            llm_response = response.choices[0].message.content
            relevance_data = json.loads(llm_response)
            print(idx, relevance_data)
            return relevance_data.get('relevance', 'No') == 'Yes'
        
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
        azure_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION") ,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # relevance_checker.eval_response(
    #     query="智邦科技股份有限公司2023年第1季的綜合損益總額是多少？",
    #     retrieved_docs=[]
    # )