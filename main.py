import argparse
import json
import logging
import os
from copy import deepcopy
from datetime import datetime

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from tqdm import tqdm

from Model.ans_generator import answer_generation
from Model.finanace_query_preprocess import query_preprocessor
from Model.search_core import (
    dense_search_with_cross_encoder,
    finance_main,
    qdrant_dense_search,
)

logging.basicConfig(
    level=logging.INFO,
    filename="log.txt",
    format="[%(asctime)s %(levelname)-8s] %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
)


def main(question_set, doc_set):
    start_time = datetime.now()

    load_dotenv()
    client = QdrantClient(url=os.getenv("qdrant_url"), timeout=60)

    ###### insurance ######
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="insurance_chunk",
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        retrieval_mode=RetrievalMode.DENSE,
    )

    insurance_data = [
        item for item in question_set["questions"] if item["category"] == "insurance"
    ]

    insurance_answers = []
    for Q in tqdm(
        insurance_data, desc=f"Processing {insurance_data[0]['category']} questions"
    ):
        insurance_search = dense_search_with_cross_encoder(
            vector_store=vector_store,
            question=Q,
            k_dense=5,
            k_cross=1,
        )

        insurance_answers.append(
            {
                "qid": Q["qid"],
                "query": Q["query"],
                "generate": answer_generation(Q, insurance_search),
                "retrieve": int(insurance_search[0].metadata["source_id"]),
                "category": insurance_search[0].metadata["category"],
            }
        )

    ###### finance #####
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="finance_recursive_chunk",
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        retrieval_mode=RetrievalMode.DENSE,
    )

    # finance_question_set = [
    #     item for item in question_set["questions"] if item["category"] == "finance"
    # ]
    # finance_question_set = query_preprocessor(finance_question_set=finance_question_set)
    # # Save the parsed query
    # time_now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    # with open(f"parsed_query_{time_now}.json", "w") as Output:
    #     json.dump(finance_question_set, Output, ensure_ascii=False, indent=4)

    with open("parsed_query_2024_11_23_16_43.json", "r") as q:
        finance_question_set = json.load(q)

    finance_answers = []
    for Q in tqdm(
        finance_question_set,
        desc=f"Processing {finance_question_set[0]['category']} questions",
    ):
        Q_copy = deepcopy(Q)
        finance_search = finance_main(vector_store, Q_copy, doc_set, score_threshold=70)
        Q["source"] = [finance_search[0].metadata["source_id"]]
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="finance_table_and_summary",
            embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
            retrieval_mode=RetrievalMode.DENSE,
        )
        finance_retrieve = qdrant_dense_search(Q, vector_store, k=1)
        finance_answers.append(
            {
                "qid": Q["qid"],
                "query": Q["query"],
                "generate": answer_generation(Q, finance_retrieve),
                "retrieve": int(finance_retrieve[0].metadata["source_id"]),
                "category": finance_retrieve[0].metadata["category"],
            }
        )

    ###### faq #####
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="qa_dense_e5",
        embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
        retrieval_mode=RetrievalMode.DENSE,
    )

    faq_data = [item for item in question_set["questions"] if item["category"] == "faq"]
    faq_answers = []
    for Q in tqdm(faq_data, desc=f"Processing {faq_data[0]['category']} questions"):
        faq_search = qdrant_dense_search(
            vector_store=vector_store,
            question=Q,
            k=1,
        )

        faq_answers.append(
            {
                "qid": Q["qid"],
                "query": Q["query"],
                "generate": answer_generation(Q, faq_search),
                "retrieve": int(faq_search[0].metadata["source_id"]),
                "category": faq_search[0].metadata["category"],
            }
        )

    answers = insurance_answers + finance_answers + faq_answers

    with open("./generation_result.json", "w") as Output:
        json.dump({"answers": answers}, Output, ensure_ascii=False, indent=4)

    end_time = datetime.now()

    print(f"Total time: {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process paths for question and finance document sets."
    )

    parser.add_argument(
        "--questions_path",
        type=str,
        default="./複賽資料集說明/questions_example.json",
        help="Path to the questions JSON file",
    )
    parser.add_argument(
        "--parsed_finance_path",
        type=str,
        default="./finance_extract_directly_patched.json",
        help="Path to the parsed finance documents JSON file",
    )

    args = parser.parse_args()

    with open(args.questions_path, "r") as q:
        question_set = json.load(q)

    with open(args.parsed_finance_path, "r") as d:
        doc_set = json.load(d)

    main(question_set, doc_set)
