import json
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from rapidfuzz import fuzz, process
from tqdm import tqdm
from pydantic import BaseModel

from .hybrid_search_rrf import dense_search_with_cross_encoder

load_dotenv()
client = QdrantClient(url=os.getenv("qdrant_url"), timeout=60)

# dense search
vector_store = QdrantVectorStore(
    client=client,
    collection_name="finance_recursive_chunk",
    embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"),
    retrieval_mode=RetrievalMode.DENSE,
)


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

class FuzzySearchEngine:
    def __init__(
        self,
        similarity_threshold: float = 50,
        score_threshold: float = 90,
        max_matches: int = 3
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
        self,
        target_ids: List[str],
        document_collection: Dict[str, str]
    ) -> pd.Series:
        """Filter documents based on target IDs."""
        target_id_set = set(str(id_) for id_ in target_ids)
        filtered_docs = {k: v for k, v in document_collection.items() 
                       if k in target_id_set}
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
                limit=None
            )

            if not matches:
                continue

            # Process top matches
            top_matches = sorted(
                matches, 
                key=lambda x: x[1], 
                reverse=True
            )[:self.max_matches]
            
            for match in top_matches:
                term_matches.append(FuzzyTermMatch(
                    search_term=term,
                    matched_text=match[0],
                    doc_id=match[2],
                    score=match[1]
                ))

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
                ["search_term", "score"],
                ascending=[True, False],
                inplace=True
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
                combined_score=(np.mean(scores) * len(scores)) / len(search_terms)
            )
            combined_results.append(result)

        combined_df = pd.DataFrame([result.dict() for result in combined_results])
        if not combined_df.empty:
            combined_df.sort_values("combined_score", ascending=False, inplace=True)

        return term_results, combined_df

    def _extract_matched_sources(
        self,
        combined_results: pd.DataFrame,
        question: QuestionDict
    ) -> List[str]:
        """Extract document sources from search results or fall back to question source."""
        if not combined_results.empty and "doc_id" in combined_results.columns:
            return [doc_id.split("_")[0] for doc_id in combined_results["doc_id"]]
        return question['source']

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
        filtered_docs = self._filter_documents_by_ids(
            question['source'],
            doc_set
        )
        search_query = pd.Series(question['parsed_query']['keyword'])

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
                    results_over_threshold,
                    question
                )
            else:
                matched_sources = [str(i) for i in question['source']]

        else:
            matched_sources = [str(i) for i in question['source']]
        
        question['source'] = matched_sources
        
        return question

def finance_main(
    vector_store: QdrantVectorStore,
    question: QuestionDict,
    doc_set: Dict[str, str],
    score_threshold: float = 90
) -> List[Document]:
    """
    Main entry point for document search and retrieval.
    """
    search_engine = FuzzySearchEngine(
        vector_store=vector_store,
        similarity_threshold=50,
        score_threshold=score_threshold,
        max_matches=3
    )
    limited_question = search_engine.search(question, doc_set)

    limited_question['query'] = limited_question['parsed_query']['scenario']

    return dense_search_with_cross_encoder(vector_store=vector_store,
        question=question,
        k_dense=5,
        k_cross=1
        )


# def main(finance_question_set, doc_set):
#     search_result = {}
#     search_list = []
#     answer = []
#     score_threshold = 90

#     for question in tqdm(finance_question_set):
#         targets = question["source"]  # get all source list from this question

#         main_string = filter_docs(targets, doc_set)  # get docs accroding to source list
#         q_keywords = question["parsed_query"]["keyword"]
#         # q_keywords.append(question['parsed_query']['season'])
#         substring = pd.Series(q_keywords)  # keywords ready to search

#         # Get both per-term and combined results
#         term_results, combined_results = advanced_fuzzy_search(
#             main_string, substring, top_n=3
#         )

#         if not combined_results.empty:
#             results_over_criteria = combined_results[
#                 combined_results.avg_score >= score_threshold
#             ]
#             if not results_over_criteria.empty:
#                 matched_source = get_matched_source(results_over_criteria, question)
#             else:
#                 matched_source = [str(i) for i in question["source"]]
#                 # matched_source = ["-1"]
#                 # print(f"qid: {question['qid']} has no avg score over 80 in fuzzy search")
#         else:
#             matched_source = [str(i) for i in question["source"]]
#             # matched_source = ["-1"]
#             # print(f"qid: {question['qid']} has no result in fuzzy search")

#         search_result.update({question["qid"]: matched_source})
#         search_list.append(
#             {
#                 "qid": question["qid"],
#                 "retrieve": [
#                     int(matched_source[i]) for i in range(len(matched_source))
#                 ],
#                 "category": question["category"],
#             }
#         )

#         # question['query'] = ';'.join(question["parsed_query"]["keyword"])
#         question["query"] = question["parsed_query"]["scenario"]
#         question["source"] = matched_source

#         query_response = dense_search_with_cross_encoder(
#             vector_store=vector_store,
#             question=question,
#             k_dense=5,
#             k_cross=1,
#         )

#         answer.append(
#             {
#                 "qid": question["qid"],
#                 "retrieve": int(query_response[0].metadata["source_id"]),
#                 "category": query_response[0].metadata["category"],
#             }
#         )

#     with open("./brute_80_dense_cross_scenario_v4.json", "w") as a:
#         json.dump({"answers": answer}, a, indent=4, ensure_ascii=False)
