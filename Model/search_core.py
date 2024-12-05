"""
Module for document retrieval and search functionality using embeddings,
vector stores, and various matching algorithms like BM25 and fuzzy matching.

This module integrates several libraries and services such as Qdrant for
vector storage, rapidfuzz for fuzzy text matching, and BM25 for ranking.
It includes a retry mechanism and defines classes and functions to facilitate
document search, filtering, and ranking operations.
"""

from copy import deepcopy
from functools import wraps
from time import sleep
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jieba
import numpy as np
import pandas as pd
import torch
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process

# 給 HuggingFace 模型吃 cuda 用
model_kwargs = {"device": "cuda"} if torch.cuda.is_available() else None


def retry(retries: int = 3, delay: float = 1):
    """
    A retry decorator that wraps a function and retries it up to a specified
    number of times with a specified delay between attempts.

    Args:
        retries (int): Number of retry attempts.
        delay (float): Delay between retries in seconds.

    Returns:
        callable: A decorator that applies retry logic to the wrapped function.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _retries, _delay = retries, delay
            for attempt in range(1, _retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < _retries:
                        print(
                            f"{func.__name__} 第 {attempt} 次嘗試失敗：{e}。等待 {_delay} 秒後重試..."
                        )
                        sleep(_delay)
                    else:
                        print(f"{func.__name__} 嘗試 {attempt} 次失敗，放棄重試。")
                        raise

        return wrapper

    return decorator


class ParsedQuery(BaseModel):
    """Represents a parsed query structure with details for searching."""

    company: str
    year: str
    season: str
    scenario: str
    keyword: List[str]


class QuestionDict(BaseModel):
    """Represents a structured question with associated metadata and parsed query information."""

    qid: str
    source: List[int]
    query: str
    category: str
    parsed_query: Optional[ParsedQuery]


class StandardizedResult(BaseModel):
    """Represents a standardized search result with source ID, score, and rank."""

    source_id: str
    score: float
    rank: int


class FuzzyTermMatch(BaseModel):
    """Represents a fuzzy match result for a search term within a document."""

    search_term: str
    matched_text: str
    doc_id: str
    score: float


class FuzzySearchResult(BaseModel):
    """Represents the result of a fuzzy search across multiple terms within a document."""

    doc_id: str
    matched_text: str
    avg_score: float
    max_score: float
    matching_terms: int
    total_terms: int
    combined_score: float


class SearchFusion:
    """
    搜尋結果融合器，使用 Reciprocal Rank Fusion (RRF) 方法將 dense vector search 和 BM25 搜尋的結果進行融合。

    Attributes:
        k (int): RRF的k參數，用於控制排名靠後結果的影響力。

    Methods:
        _convert_dense_results: 轉換 dense search 的結果為標準格式。
        _convert_bm25_results: 轉換 BM25 的結果為標準格式。
        reciprocal_rank_fusion: 使用 RRF 方法融合 dense search 和 BM25 搜尋的結果。
    """

    def __init__(self, k: int = 60):
        """
        初始化搜尋結果融合器
        Args:
            k: RRF的k參數，用於控制排名靠後結果的影響力
        """
        self.k = k

    def _convert_dense_results(
        self, dense_results: List[dict]
    ) -> List[StandardizedResult]:
        """
        轉換dense search的結果為標準格式
        """
        results = []
        for rank, result in enumerate(dense_results):
            results.append(
                StandardizedResult(
                    source_id=result.metadata["source_id"],
                    score=1.0 / (rank + 1),
                    rank=rank,
                )
            )
        return results

    def _convert_bm25_results(
        self, bm25_results: List[str]
    ) -> List[StandardizedResult]:
        """
        轉換BM25的結果為標準格式
        """
        return [
            StandardizedResult(source_id=source_id, score=1.0 / (rank + 1), rank=rank)
            for rank, source_id in enumerate(bm25_results)
        ]

    def reciprocal_rank_fusion(
        self,
        dense_results: List[dict],
        bm25_results: List[str],
        return_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        使用RRF方法融合兩種搜尋結果
        Args:
            dense_results: dense vector search的結果列表
            bm25_results: BM25搜尋的結果ID列表
            return_scores: 是否返回文檔的分數
        Returns:
            融合後排序的文檔ID列表，或者文檔ID和分數的元組列表
        """
        # 轉換結果為標準格式
        dense_standard = self._convert_dense_results(dense_results)
        bm25_standard = self._convert_bm25_results(bm25_results)

        # 合併所有唯一的文檔ID
        all_docs = set(r.source_id for r in dense_standard + bm25_standard)

        # 計算每個文檔的RRF分數
        rrf_scores = {}
        for source_id in all_docs:
            rrf_score = 0

            # 在dense結果中尋找排名
            dense_rank = next(
                (r.rank for r in dense_standard if r.source_id == source_id), None
            )
            if dense_rank is not None:
                rrf_score += 1 / (self.k + dense_rank + 1)

            # 在BM25結果中尋找排名
            bm25_rank = next(
                (r.rank for r in bm25_standard if r.source_id == source_id), None
            )
            if bm25_rank is not None:
                rrf_score += 1 / (self.k + bm25_rank + 1)

            rrf_scores[source_id] = rrf_score

        # 根據RRF分數排序文檔
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        if return_scores:
            return sorted_docs
        return [source_id for source_id, _ in sorted_docs]


class FuzzySearchEngine:
    """
    Fuzzy Search Engine for finding matches in a document collection based on similarity thresholds.

    Attributes:
        similarity_threshold (float): Minimum similarity score for considering a fuzzy match.
        score_threshold (float): Threshold for filtering final search scores.
        max_matches (int): Maximum number of matches to return for each search term.

    Methods:
        _filter_documents_by_ids: Filters documents by a specified list of IDs.
        _extract_matched_sources: Extracts matched document sources based on combined search results.
        fuzzy_search: Conducts a fuzzy search on documents using multiple search terms.
        search: Performs the main search process, combining fuzzy matching with additional filters.
    """

    def __init__(
        self,
        similarity_threshold: float = 50,
        score_threshold: float = 90,
        max_matches: int = 3,
    ):
        """
        Initializes the fuzzy search engine with specified parameters.

        Args:
            similarity_threshold (float): Minimum similarity score for fuzzy matching.
            score_threshold (float): Threshold for final score filtering.
            max_matches (int): Maximum number of matches to return per search term.
        """
        self.similarity_threshold = similarity_threshold
        self.score_threshold = score_threshold
        self.max_matches = max_matches

    def _filter_documents_by_ids(
        self, target_ids: List[str], document_collection: Dict[str, str]
    ) -> pd.Series:
        """
        Filters documents based on specified target IDs.

        Args:
            target_ids (List[str]): List of document IDs to filter.
            document_collection (Dict[str, str]): Dictionary of all documents with IDs as keys.

        Returns:
            pd.Series: Series containing filtered documents with the specified IDs.
        """
        target_id_set = set(str(id_) for id_ in target_ids)
        filtered_docs = {
            k: v for k, v in document_collection.items() if k in target_id_set
        }
        return pd.Series(filtered_docs)

    def _extract_matched_sources(
        self, combined_results: pd.DataFrame, question: QuestionDict
    ) -> List[str]:
        """
        Extracts document sources from search results or falls back to the question source.

        Args:
            combined_results (pd.DataFrame): DataFrame with search results.
            question (QuestionDict): Dictionary containing the search question and metadata.

        Returns:
            List[str]: List of document source IDs.
        """
        if not combined_results.empty and "doc_id" in combined_results.columns:
            return [doc_id.split("_")[0] for doc_id in combined_results["doc_id"]]
        return question["source"]

    def fuzzy_search(
        self,
        documents: pd.Series,
        search_terms: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs a fuzzy search across documents with multiple search terms, calculating scores for each term.

        Args:
            documents (pd.Series): Series of documents to search.
            search_terms (pd.Series): Series of search terms.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing individual term matches and combined results.
        """
        term_matches = []
        document_scores: Dict[str, List[float]] = {}

        # Search for each term
        for term in search_terms:
            matches = process.extract(
                term,
                documents,
                scorer=fuzz.partial_ratio,
                score_cutoff=self.similarity_threshold,
                limit=None,
            )

            if not matches:
                continue

            # Process top matches
            top_matches = sorted(matches, key=lambda x: x[1], reverse=True)[
                : self.max_matches
            ]

            for match in top_matches:
                term_matches.append(
                    FuzzyTermMatch(
                        search_term=term,
                        matched_text=match[0],
                        doc_id=match[2],
                        score=match[1],
                    )
                )

            # Store all scores for combined scoring
            for match in matches:
                doc_id = match[2]
                score = match[1]
                if doc_id not in document_scores:
                    document_scores[doc_id] = []
                document_scores[doc_id].append(score)

        # Create DataFrames
        term_results = pd.DataFrame([match.dict() for match in term_matches])
        if not term_results.empty:
            term_results.sort_values(
                ["search_term", "score"], ascending=[True, False], inplace=True
            )

        # Calculate combined scores
        combined_results = []
        for doc_id, scores in document_scores.items():
            result = FuzzySearchResult(
                doc_id=doc_id,
                matched_text=documents[doc_id],
                avg_score=np.mean(scores),
                max_score=max(scores),
                matching_terms=len(scores),
                total_terms=len(search_terms),
                combined_score=(np.mean(scores) * len(scores)) / len(search_terms),
            )
            combined_results.append(result)

        combined_df = pd.DataFrame([result.dict() for result in combined_results])
        if not combined_df.empty:
            combined_df.sort_values("combined_score", ascending=False, inplace=True)

        return term_results, combined_df

    def search(
        self,
        question: QuestionDict,
        doc_set: Dict[str, str],
    ) -> QuestionDict:
        """
        Main search method that combines fuzzy search and document filtering.

        Args:
            question (QuestionDict): Dictionary containing the search question with parsed query and metadata.
            doc_set (Dict[str, str]): Dictionary of documents with IDs as keys.

        Returns:
            QuestionDict: Updated question dictionary with search results populated in the source.
        """
        # Filter and prepare documents
        filtered_docs = self._filter_documents_by_ids(question["source"], doc_set)
        search_query = pd.Series(question["parsed_query"]["keyword"])

        # Perform fuzzy search
        term_results, combined_results = self.fuzzy_search(
            filtered_docs,
            search_query,
        )

        # Process results
        if not combined_results.empty:
            results_over_threshold = combined_results[
                combined_results.avg_score >= self.score_threshold
            ]
            if not results_over_threshold.empty:
                matched_sources = self._extract_matched_sources(
                    results_over_threshold, question
                )
            else:
                matched_sources = [str(i) for i in question["source"]]

        else:
            matched_sources = [str(i) for i in question["source"]]

        output_q = deepcopy(question)
        output_q["source"] = matched_sources

        return output_q

    def search_get_text(
        self, question: QuestionDict, doc_set: Dict[str, str]
    ) -> Optional[str]:
        search_query = pd.Series(question["parsed_query"]["keyword"])
        term_results, combined_results = self.fuzzy_search(
            doc_set,
            search_query,
        )

        return term_results["matched_text"] if not term_results.empty else None


# DENSE SEARCH
@retry(retries=3, delay=1)
def qdrant_dense_search(
    question: QuestionDict, vector_store: QdrantVectorStore, k: int = 3
) -> List[Document]:
    """
    Perform dense search using Qdrant

    Args:
        question: Search question containing query and metadata
        vector_store: QdrantVectorStore instance to perform search on
        k: Number of results to return

    Returns:
        List of Document objects containing search results
    """

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

    return vector_store.similarity_search(
        question["query"], filter=filter_conditions, k=k
    )


# BM25 SEARCH
def bm25_jieba_search(
    question: QuestionDict, corpus_dict: dict, k: int = 3
) -> List[str]:
    """
    Perform BM25 search using jieba tokenization on a given corpus and query.

    Args:
        question: Search question containing query and metadata
        corpus_dict: Dictionary mapping file names to text content
        k: Number of results to return

    Returns:
        List of document IDs that are the top k results
    """
    filtered_corpus = [corpus_dict[str(file)] for file in question["source"]]
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut_for_search(question["query"]))
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=k)
    result = []
    for a in ans:
        keys = [key for key, value in corpus_dict.items() if value == a]
        result.extend(keys)

    return result


def hybrid_search_rerank(
    question: QuestionDict,
    vector_store: QdrantVectorStore,
    corpus_dict: dict,
    k: int = 1,
) -> dict:
    """
    Hybrid search method that combines dense search and BM25 search.

    Performs a dense search using Qdrant and a BM25 search using jieba tokenization.
    The results are then fused using a SearchFusion instance with Reciprocal Rank Fusion (RRF).
    The final results are returned as a dictionary with the qid and retrieve list.

    Args:
        question (QuestionDict): Dictionary containing the search question with parsed query and metadata.
        vector_store (QdrantVectorStore): QdrantVectorStore instance to perform search on.
        corpus_dict (dict): Dictionary mapping file names to text content.
        k (int, optional): Number of results to return. Defaults to 1.

    Returns:
        dict: Dictionary containing the qid and retrieve list.
    """
    dense_results = qdrant_dense_search(question, vector_store, k=3)
    bm25_result = bm25_jieba_search(question, corpus_dict, k=3)

    # 融合結果
    fusion = SearchFusion(k=60)
    final_results = fusion.reciprocal_rank_fusion(dense_results, bm25_result)

    # 返回結果列表
    if final_results:
        return {
            "qid": question["qid"],
            "retrieve": [
                int(final_results[i]) for i in range(min(k, len(final_results)))
            ],
            "category": question["category"],
        }
    else:
        return {"qid": question["qid"], "retrieve": [-1], "category": "not found"}


@retry(retries=3, delay=1)
def crosss_encoder_rerank(
    question: QuestionDict,
    documents: Sequence[Document],
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
    k: int = 1,
) -> Sequence[Document]:
    """
    使用指定的 Cross-Encoder 模型對文件進行重新排序。

    Args:
        question (dict): 包含查詢問題的字典，'query' 字段應包含查詢的內容。
        documents (Sequence[Document]): 待排序的文件列表。
        cross_encoder_model (str, optional): 使用的 Cross-Encoder 模型名稱。預設為 'BAAI/bge-reranker-v2-m3'。
        k (int, optional): 返回的前 k 個文件，預設為 1。

    Returns:
        Sequence[Document]: 經過模型重新排序後的文件列表。
    """
    model = HuggingFaceCrossEncoder(
        model_name=cross_encoder_model, model_kwargs=model_kwargs
    )
    compressor = CrossEncoderReranker(model=model, top_n=k)
    return compressor.compress_documents(documents=documents, query=question["query"])


def finance_main(
    vector_store: QdrantVectorStore,
    question: QuestionDict,
    doc_set: Dict[str, str],
    score_threshold: float = 90,
) -> List[Document]:
    """
    finance 的搜尋引擎，首先使用 fuzzy search 來初步 filter 文件，然後使用 dense vector search with cross encoder 進一步限縮範圍。

    Args:
        vector_store (QdrantVectorStore): Qdrant 的向量庫實例。
        question (QuestionDict): 包含查詢問題的字典，'query' 字段應包含查詢的內容。
        doc_set (Dict[str, str]): 文件的 ID 到內容的映射。
        score_threshold (float, optional):  Fuzzy search 的分數門檻，預設為 90。

    Returns:
        List[Document]: 經過排序後的文件列表。
    """
    search_engine = FuzzySearchEngine(
        similarity_threshold=50, score_threshold=score_threshold, max_matches=3
    )
    fuzzy_result = deepcopy(search_engine.search(question, doc_set))
    fuzzy_result["query"] = fuzzy_result["parsed_query"]["scenario"]
    return dense_search_with_cross_encoder(
        vector_store=vector_store, question=fuzzy_result, k_dense=3, k_cross=1
    )


def dense_search_with_cross_encoder(
    vector_store: QdrantVectorStore,
    question: QuestionDict,
    k_dense: int,
    k_cross: int = 1,
) -> List[Document]:
    """
    Perform a dense search using Qdrant and rerank the results with a Cross-Encoder model.

    Args:
        vector_store (QdrantVectorStore): The vector store used to perform the dense search.
        question (QuestionDict): Dictionary containing the search question with parsed query and metadata.
        k_dense (int): Number of top dense search results to retrieve.
        k_cross (int, optional): Number of top results to return after Cross-Encoder reranking. Defaults to 1.

    Returns:
        List[Document]: The list of documents after reranking with the Cross-Encoder.
    """

    dense_result = qdrant_dense_search(
        vector_store=vector_store, question=question, k=k_dense
    )
    return crosss_encoder_rerank(question=question, documents=dense_result, k=k_cross)


@retry(retries=3, delay=1)
def retrieve_document_by_source_ids(
    client: QdrantClient, collection_name: str, source_ids: List[int]
) -> List[Document]:
    """
    從 Qdrant 擷取指定 source_id 的文件並轉換成 LangChain Document 格式
    Args:
        client: Qdrant client 實例
        collection_name: 集合名稱
        source_ids: 要擷取的文件 source_id 列表
    Returns:
        List[Document]: LangChain Document 物件列表
    """
    # 建立 filter 條件，使用 must 確保完全符合指定的 source_ids
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="metadata.source_id",
                match=MatchAny(any=[str(i) for i in source_ids]),
            )
        ]
    )

    # 使用 scroll 方法取得符合條件的所有文件
    get_result = client.scroll(
        collection_name=collection_name,
        scroll_filter=filter_condition,
        limit=len(source_ids),
    )

    # 將 Qdrant 的搜尋結果轉換成 LangChain Document 格式
    documents = []
    for record in get_result[0]:  # search_result[0] 包含實際的記錄
        # 創建 Document 物件，保留所有 metadata
        doc = Document(
            page_content=record.payload.get(
                "page_content", ""
            ),  # 假設文本內容存儲在 'text' 欄位
            metadata=record.payload.get("metadata", {}),
        )
        documents.append(doc)

    return documents


if __name__ == "__main__":
    print("本模組是用於實現文件檢索與融合搜尋功能的工具包，請將其匯入至專案中使用。")
    print("\n如果需要測試模組功能，建議撰寫自己的測試腳本來驗證特定功能。")
