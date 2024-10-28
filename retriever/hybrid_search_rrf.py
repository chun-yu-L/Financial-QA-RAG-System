import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Union

import jieba
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue
from rank_bm25 import BM25Okapi


@dataclass
class SearchResult:
    source_id: str
    score: float
    rank: int


class SearchFusion:
    def __init__(self, k: int = 60):
        """
        初始化搜尋結果融合器
        Args:
            k: RRF的k參數，用於控制排名靠後結果的影響力
        """
        self.k = k

    def _convert_dense_results(self, dense_results: List[dict]) -> List[SearchResult]:
        """
        轉換dense search的結果為標準格式
        """
        results = []
        for rank, result in enumerate(dense_results):
            results.append(
                SearchResult(
                    source_id=result.metadata["source_id"],
                    score=1.0 / (rank + 1),
                    rank=rank,
                )
            )
        return results

    def _convert_bm25_results(self, bm25_results: List[str]) -> List[SearchResult]:
        """
        轉換BM25的結果為標準格式
        """
        return [
            SearchResult(source_id=source_id, score=1.0 / (rank + 1), rank=rank)
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


# DENSE SEARCH
def qdrant_dense_search(
    question: dict, vector_store: QdrantVectorStore, k=3
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
def bm25_jieba_search(question: dict, corpus_dict: dict, k=3) -> List[str]:
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
    question: dict, vector_store: QdrantVectorStore, corpus_dict: dict
):
    dense_results = qdrant_dense_search(question, vector_store, k=3)
    bm25_result = bm25_jieba_search(question, corpus_dict, k=3)

    # 融合結果
    fusion = SearchFusion(k=60)
    final_results = fusion.reciprocal_rank_fusion(dense_results, bm25_result)

    # 返回第一個結果
    if final_results:
        return {
            "qid": question["qid"],
            "retrieve": int(final_results[0]),
            "category": question["category"],
        }
    else:
        return {"qid": question["qid"], "retrieve": -1, "category": "not found"}


if __name__ == "__main__":
    load_dotenv()
    client = QdrantClient(url=os.getenv("qdrant_url"), timeout=60)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="insurance_hybrid_bgeNbm42",
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
        retrieval_mode=RetrievalMode.DENSE,
    )

    with open("./競賽資料集/dataset/preliminary/questions_example.json", "r") as q:
        question_set = json.load(q)
    insurance_data = [
        item for item in question_set["questions"] if item["category"] == "insurance"
    ]

    with open("insurance_fulltext/corpus_insurance.json", "r") as q:
        corpus_dict = json.load(q)

    answers = []

    for Q in insurance_data:
        answers.append(hybrid_search_rerank(Q, vector_store, corpus_dict))

    with open("./09_insurance_hybrid_rrf.json", "w") as Output:
        json.dump({"answers": answers}, Output, ensure_ascii=False, indent=4)
