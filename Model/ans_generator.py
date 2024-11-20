from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue


def get_qa_answer(question, llm, client):
    ## retrieve
    filter_conditions = Filter(
        must=[
            FieldCondition(
                key="metadata.category",
                match=MatchValue(value=question["category"]),
            ),
            FieldCondition(
                key="metadata.source_id",
                match=MatchAny(any=[str(i) for i in question["source"]]),
            ),
        ]
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="qa_dense_e5",
        embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
        retrieval_mode=RetrievalMode.DENSE,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"filter": filter_conditions, "k": 1}
    )

    ## prompt
    faq_prompt = ChatPromptTemplate(
        [
            (
                "human",
                """
                你是一位精通金融問答的專家，請嚴格地根據<參考資料>回答問題，使用正體中文

                問題：{question}

                <參考資料>
                {context}
                </參考資料>

                專家回答：
                """,
            )
        ]
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | faq_prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"question": question["query"]})
