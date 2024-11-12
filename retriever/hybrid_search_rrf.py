import json
import os
from functools import wraps
from time import sleep
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jieba
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from tqdm import tqdm


def retry(retries: int = 3, delay: float = 1):
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
    company: str
    year: str
    season: str
    scenario: str
    keyword: List[str]


class QuestionDict(BaseModel):
    qid: str
    source: List[int]
    query: str
    category: str
    parsed_query: Optional[ParsedQuery]


class StandardizedResult(BaseModel):
    source_id: str
    score: float
    rank: int


class FuzzyTermMatch(BaseModel):
    search_term: str
    matched_text: str
    doc_id: str
    score: float


class FuzzySearchResult(BaseModel):
    doc_id: str
    matched_text: str
    avg_score: float
    max_score: float
    matching_terms: int
    total_terms: int
    combined_score: float


class SearchFusion:
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
    def __init__(
        self,
        similarity_threshold: float = 50,
        score_threshold: float = 90,
        max_matches: int = 3,
    ):
        """
        Initialize the search engine.

        Args:
            similarity_threshold: Minimum similarity score for fuzzy matching
            score_threshold: Threshold for final score filtering
            max_matches: Maximum number of matches to return per term
        """
        self.similarity_threshold = similarity_threshold
        self.score_threshold = score_threshold
        self.max_matches = max_matches

    def _filter_documents_by_ids(
        self, target_ids: List[str], document_collection: Dict[str, str]
    ) -> pd.Series:
        """Filter documents based on target IDs."""
        target_id_set = set(str(id_) for id_ in target_ids)
        filtered_docs = {
            k: v for k, v in document_collection.items() if k in target_id_set
        }
        return pd.Series(filtered_docs)

    def fuzzy_search(
        self,
        documents: pd.Series,
        search_terms: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform fuzzy search across documents with multiple search terms."""
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

    def _extract_matched_sources(
        self, combined_results: pd.DataFrame, question: QuestionDict
    ) -> List[str]:
        """Extract document sources from search results or fall back to question source."""
        if not combined_results.empty and "doc_id" in combined_results.columns:
            return [doc_id.split("_")[0] for doc_id in combined_results["doc_id"]]
        return question["source"]

    def search(
        self,
        question: QuestionDict,
        doc_set: Dict[str, str],
    ) -> QuestionDict:
        """
        Main search method combining fuzzy and dense search.

        Args:
            question: Search question containing query and metadata
            doc_set: Collection of documents to search in

        Returns:
            List of search results
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

        question["source"] = matched_sources

        return question


# DENSE SEARCH
@retry(retries=3, delay=1)
def qdrant_dense_search(
    question: QuestionDict, vector_store: QdrantVectorStore, k: int = 3
) -> List[Document]:
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
    model = HuggingFaceCrossEncoder(model_name=cross_encoder_model)
    compressor = CrossEncoderReranker(model=model, top_n=k)
    return compressor.compress_documents(documents=documents, query=question["query"])


def finance_main(
    vector_store: QdrantVectorStore,
    question: QuestionDict,
    doc_set: Dict[str, str],
    score_threshold: float = 90,
) -> List[Document]:
    """
    Main entry point for document search and retrieval.
    """
    search_engine = FuzzySearchEngine(
        similarity_threshold=50, score_threshold=score_threshold, max_matches=3
    )
    limited_question = search_engine.search(question, doc_set)

    limited_question["query"] = limited_question["parsed_query"]["scenario"]

    return dense_search_with_cross_encoder(
        vector_store=vector_store, question=question, k_dense=5, k_cross=1
    )


def dense_search_with_cross_encoder(
    vector_store: QdrantVectorStore,
    question: QuestionDict,
    k_dense: int,
    k_cross: int = 1,
) -> List[Document]:
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
    faq_vector_store = QdrantVectorStore(
        client=client,
        collection_name="qa_dense_e5",
        embedding=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
        retrieval_mode=RetrievalMode.DENSE,
    )

    # insurance
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

    # faq
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

    answers = insurance_answers + faq_answers

    with open("./13_test.json", "w") as Output:
        json.dump({"answers": answers}, Output, ensure_ascii=False, indent=4)
