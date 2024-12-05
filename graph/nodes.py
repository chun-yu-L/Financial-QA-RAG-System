from typing import Any, Dict, Optional

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from graph.state import QAState
from Model.ans_generator import answer_generation
from Model.finanace_query_preprocess import query_preprocessor
from Model.search_core import (
    FuzzySearchEngine,
    dense_search_with_cross_encoder,
    finance_main,
    qdrant_dense_search,
    retrieve_document_by_source_ids,
)
from Model.utils import count_tokens


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
        model_kwargs = {"device": "cuda"} if torch.cuda.is_available() else {}

    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


def route_question(state: QAState) -> str:
    routing_map = {
        "insurance": "process_insurance",
        "finance": "process_finance",
        "faq": "process_faq",
    }

    category = state["question"]["category"]
    if category not in routing_map:
        raise ValueError(f"Unsupported category: {category}")

    return routing_map[category]


def insurance_node(state: QAState) -> QAState:
    embedding_model = create_embedding_model(model_name="BAAI/bge-m3")
    vector_store = QdrantVectorStore(
        client=state["client"],
        collection_name="insurance_chunk",
        embedding=embedding_model,
        retrieval_mode=RetrievalMode.DENSE,
    )

    Q = state["question"]
    insurance_search = dense_search_with_cross_encoder(
        vector_store=vector_store,
        question=Q,
        k_dense=3,
        k_cross=1,
    )

    del embedding_model
    torch.cuda.empty_cache()

    state["answer"] = {
        "qid": Q["qid"],
        "query": Q["query"],
        "generate": answer_generation(Q, insurance_search),
        "retrieve": int(insurance_search[0].metadata["source_id"]),
        "category": insurance_search[0].metadata["category"],
    }
    return state


def finance_node(state: QAState) -> QAState:
    # query 預處理
    Q = query_preprocessor(finance_question_set=[state["question"]])[0]

    torch.cuda.empty_cache()

    embedding_model = create_embedding_model(model_name="BAAI/bge-m3")

    vector_store_chunk = QdrantVectorStore(
        client=state["client"],
        collection_name="finance_recursive_chunk_1500",
        embedding=embedding_model,
        retrieval_mode=RetrievalMode.DENSE,
    )

    vector_store_table = QdrantVectorStore(
        client=state["client"],
        collection_name="finance_4o_extraction",
        embedding=embedding_model,
        retrieval_mode=RetrievalMode.DENSE,
    )

    finance_search = finance_main(
        vector_store_chunk, Q, state["doc_set"], score_threshold=70
    )

    Q_copy = Q.copy()
    Q_copy["source"] = [finance_search[0].metadata["source_id"]]

    #    finance_retrieve = qdrant_dense_search(Q_copy, vector_store_table, k=3)

    del embedding_model
    torch.cuda.empty_cache()

    retrieve_doc = retrieve_document_by_source_ids(
        client=state["client"],
        collection_name="finance_4o_extraction",
        source_ids=Q_copy["source"],
    )

    state["doc_set"] = {
        item.metadata["page"]: item.page_content for item in retrieve_doc
    }

    search_engine = FuzzySearchEngine(
        similarity_threshold=50, score_threshold=80, max_matches=3
    )

    fuzzy_retrieve = search_engine.search_get_text(Q_copy, state["doc_set"])

    if fuzzy_retrieve is not None and count_tokens(fuzzy_retrieve) < 6000:
        finance_retrieve = fuzzy_retrieve
    else:
        finance_retrieve = qdrant_dense_search(Q_copy, vector_store_table, k=3)

    state["answer"] = {
        "qid": Q_copy["qid"],
        "query": Q_copy["query"],
        "generate": answer_generation(Q_copy, finance_retrieve),
        "retrieve": int(Q_copy["source"][0]),
        "category": Q_copy["category"],
    }
    return state


def faq_node(state: QAState) -> QAState:
    embedding_model = create_embedding_model(
        model_name="intfloat/multilingual-e5-large"
    )

    vector_store = QdrantVectorStore(
        client=state["client"],
        collection_name="qa_dense_e5",
        embedding=embedding_model,
        retrieval_mode=RetrievalMode.DENSE,
    )

    Q = state["question"]
    faq_search = qdrant_dense_search(
        vector_store=vector_store,
        question=Q,
        k=1,
    )

    del embedding_model
    torch.cuda.empty_cache()

    state["answer"] = {
        "qid": Q["qid"],
        "query": Q["query"],
        "generate": answer_generation(Q, faq_search),
        "retrieve": int(faq_search[0].metadata["source_id"]),
        "category": faq_search[0].metadata["category"],
    }
    return state
