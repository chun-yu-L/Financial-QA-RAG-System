import argparse
import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
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


def create_embedding_model(
    model_name: str = "BAAI/bge-m3", model_kwargs: Optional[Dict[str, Any]] = None
) -> HuggingFaceEmbeddings:
    """
    Create and return a HuggingFaceEmbeddings instance with optional default settings.

    Args:
        model_name (str): Name of the embedding model. Default is "BAAI/bge-m3".
        model_kwargs (Optional[Dict[str, Any]]): Model configuration. If None, it defaults to {"device": "cuda"} if CUDA is available.

    Returns:
        HuggingFaceEmbeddings: Preloaded embedding model instance.
    """
    if model_kwargs is None:
        model_kwargs = {"device": "cuda"} if torch.cuda.is_available() else None

    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


def process_insurance_questions(
    question_set, client: QdrantClient, embedding_model
) -> List[Dict[str, Any]]:
    """
    Process insurance-related questions

    Args:
        question_set: Dictionary containing questions
        client: Qdrant client
        embedding_model (HuggingFaceEmbeddings): Preloaded embedding model instance

    Returns:
        List of processed insurance answers
    """
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="insurance_chunk",
        embedding=embedding_model,
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
            k_dense=3,
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

    return insurance_answers


def process_finance_questions(
    question_set, doc_set, client: QdrantClient, embedding_model
) -> List[Dict[str, Any]]:
    """
    Process finance-related questions

    Args:
        question_set: Dictionary containing questions
        doc_set: Document set
        client: Qdrant client
        embedding_model (HuggingFaceEmbeddings): Preloaded embedding model instance

    Returns:
        List of processed finance answers
    """
    # Recursive chunk vector store
    vector_store_chunk = QdrantVectorStore(
        client=client,
        collection_name="finance_recursive_chunk_1500",
        embedding=embedding_model,
        retrieval_mode=RetrievalMode.DENSE,
    )

    # Table and summary vector store
    vector_store_table = QdrantVectorStore(
        client=client,
        collection_name="finance_table_and_summary",
        embedding=embedding_model,
        retrieval_mode=RetrievalMode.DENSE,
    )

    finance_question_set = [
        item for item in question_set["questions"] if item["category"] == "finance"
    ]
    finance_question_set = query_preprocessor(finance_question_set=finance_question_set)

    # Save the parsed query
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    with open(f"parsed_query_{time_now}.json", "w") as Output:
        json.dump(finance_question_set, Output, ensure_ascii=False, indent=4)

    finance_answers = []
    for Q in tqdm(
        finance_question_set,
        desc=f"Processing {finance_question_set[0]['category']} questions",
    ):
        finance_search = finance_main(
            vector_store_chunk, Q, doc_set, score_threshold=70
        )
        Q_copy = deepcopy(Q)
        Q_copy["source"] = [finance_search[0].metadata["source_id"]]

        finance_retrieve_table = qdrant_dense_search(Q_copy, vector_store_table, k=1)
        finance_retrieve_intext = qdrant_dense_search(Q_copy, vector_store_chunk, k=1)
        finance_retrieve = finance_retrieve_table + finance_retrieve_intext

        finance_answers.append(
            {
                "qid": Q_copy["qid"],
                "query": Q_copy["query"],
                "generate": answer_generation(Q_copy, finance_retrieve),
                "retrieve": int(finance_retrieve[0].metadata["source_id"]),
                "category": finance_retrieve[0].metadata["category"],
            }
        )

    return finance_answers


def process_faq_questions(question_set, client: QdrantClient, embedding_model):
    """
    Process FAQ-related questions

    Args:
        question_set: Dictionary containing questions
        client: Qdrant client
        embedding_model (HuggingFaceEmbeddings): Preloaded embedding model instance

    Returns:
        List of processed FAQ answers
    """
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="qa_dense_e5",
        embedding=embedding_model,
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

    return faq_answers


def main(question_set, doc_set):
    start_time = datetime.now()

    load_dotenv()
    client = QdrantClient(url=os.getenv("qdrant_url"), timeout=30)

    ## insurance & finance
    embedding_model_m3 = create_embedding_model(model_name="BAAI/bge-m3")

    insurance_answers = process_insurance_questions(
        question_set, doc_set, client, embedding_model_m3
    )

    finance_answers = process_finance_questions(
        question_set, doc_set, client, embedding_model_m3
    )

    del embedding_model_m3
    torch.cuda.empty_cache()

    ## faq
    embedding_model_e5 = create_embedding_model(
        model_name="intfloat/multilingual-e5-large"
    )
    faq_answers = process_faq_questions(question_set, client, embedding_model_e5)

    # concate and save
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
