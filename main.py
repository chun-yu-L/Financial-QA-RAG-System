import json
import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from tqdm import tqdm

from retriever.finanace_query_preprocess import query_preprocessor
from retriever.finance_search import finance_main
from retriever.hybrid_search_rrf import (
    dense_search_with_cross_encoder,
    qdrant_dense_search,
)


def main():
    with open("./競賽資料集/dataset/preliminary/questions_example.json", "r") as q:
        question_set = json.load(q)

    # qdrant vector store for different categories
    load_dotenv()
    client = QdrantClient(url=os.getenv("qdrant_url"), timeout=60)
    insurance_vector_store = QdrantVectorStore(
        client=client,
        collection_name="insurance_chunk",
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        retrieval_mode=RetrievalMode.DENSE,
    )
    finance_vector_store = QdrantVectorStore(
        client=client,
        collection_name="finance_recursive_chunk",
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        retrieval_mode=RetrievalMode.DENSE,
    )
    faq_vector_store = QdrantVectorStore(
        client=client,
        collection_name="qa_dense_e5",
        embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
        retrieval_mode=RetrievalMode.DENSE,
    )

    ###### insurance ######
    insurance_data = [
        item for item in question_set["questions"] if item["category"] == "insurance"
    ]

    insurance_answers = []
    for Q in tqdm(
        insurance_data, desc=f"Processing {insurance_data[0]['category']} questions"
    ):
        insurance_search = dense_search_with_cross_encoder(
            vector_store=insurance_vector_store,
            question=Q,
            k_dense=5,
        )

        insurance_answers.append(
            {
                "qid": Q["qid"],
                "retrieve": int(insurance_search[0].metadata["source_id"]),
                "category": insurance_search[0].metadata["category"],
            }
        )

    ###### finance #####
    finance_question_set = [
        item for item in question_set["questions"] if item["category"] == "finance"
    ]
    finance_question_set = query_preprocessor(finance_question_set=finance_question_set)

    with open("./finance_extract_directly_patched.json", "r") as q:
        doc_set = json.load(q)

    finance_answers = []
    for Q in tqdm(
        finance_question_set,
        desc=f"Processing {finance_question_set[0]['category']} questions",
    ):
        finance_search = finance_main(finance_vector_store, Q, doc_set)
        finance_answers.append(
            {
                "qid": Q["qid"],
                "retrieve": int(finance_search[0].metadata["source_id"]),
                "category": finance_search[0].metadata["category"],
            }
        )

    ###### faq #####
    faq_data = [item for item in question_set["questions"] if item["category"] == "faq"]
    faq_answers = []
    for Q in tqdm(faq_data, desc=f"Processing {faq_data[0]['category']} questions"):
        faq_search = qdrant_dense_search(
            vector_store=faq_vector_store,
            question=Q,
            k=1,
        )

        faq_answers.append(
            {
                "qid": Q["qid"],
                "retrieve": int(faq_search[0].metadata["source_id"]),
                "category": faq_search[0].metadata["category"],
            }
        )

    answers = insurance_answers + finance_answers + faq_answers

    with open("./retrieval_result.json", "w") as Output:
        json.dump({"answers": answers}, Output, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
