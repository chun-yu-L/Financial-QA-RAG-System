from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from Model.prompts import PROMPTS
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_qa_answer(question, llm, vector_store):
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

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"filter": filter_conditions, "k": 1}
    )

    ## prompt
    prompt = PROMPTS["faq_ans"]

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question["query"])
