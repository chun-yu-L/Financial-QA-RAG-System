import json
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import jieba
from dotenv import load_dotenv
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue
from rank_bm25 import BM25Okapi
from tqdm import tqdm


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
    question: dict, vector_store: QdrantVectorStore, k: int = 3
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
def bm25_jieba_search(question: dict, corpus_dict: dict, k: int = 3) -> List[str]:
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
    question: dict, vector_store: QdrantVectorStore, corpus_dict: dict, k: int = 1
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


def crosss_encoder_rerank(
    question: dict,
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

    for Q in tqdm(insurance_data, desc="Processing questions"):
        hybrid_rank_result = hybrid_search_rerank(Q, vector_store, corpus_dict)
        hybrid_docs = retrieve_document_by_source_ids(
            client=client,
            collection_name="insurance_hybrid_bgeNbm42",
            source_ids=hybrid_rank_result["retrieve"],
        )
        encoder_rerank = crosss_encoder_rerank(Q, hybrid_docs)

        if encoder_rerank:
            query_result = encoder_rerank[0].metadata
        else:
            print(f"Find no answer for {Q['qid']}")
            query_result = {
                "category": "not found",
                "file_source": "not found",
                "page_number": -1,
                "source": "-1",
                "source_id": "-1",
            }

        answers.append(
            {
                "qid": Q["qid"],
                "retrieve": int(query_result["source_id"]),
                "category": query_result["category"],
            }
        )

    with open("./09_insurance_hybrid_rrf.json", "w") as Output:
        json.dump({"answers": answers}, Output, ensure_ascii=False, indent=4)
