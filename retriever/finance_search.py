import json
import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from rapidfuzz import fuzz, process
from tqdm import tqdm

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


def filter_docs(targets, doc_set):
    target_strings = set(str(t) for t in targets)

    filtered_docs = {k: v for k, v in doc_set.items() if k in target_strings}

    return pd.Series(filtered_docs)


def advanced_fuzzy_search(main_series, search_terms, score_cutoff=50, top_n=3):
    """
    Perform advanced fuzzy search with top N scores per term and combined scoring.

    Parameters:
    - main_series: pd.Series of documents to search in
    - search_terms: pd.Series of terms to search for
    - score_cutoff: minimum similarity score (0-100)
    - top_n: number of top scores to keep per search term

    Returns:
    - per_term_results: DataFrame with top N matches per search term
    - combined_scores: Series with combined scores for each document
    """
    per_term_results = []
    all_scores = {}  # For tracking all scores per document

    # Process each search term
    for term in search_terms:
        # Get all matches above score_cutoff
        matches = process.extract(
            term,
            main_series,
            scorer=fuzz.partial_ratio,
            score_cutoff=score_cutoff,
            limit=None,
        )

        if matches:
            # Sort by score and get top N
            top_matches = sorted(matches, key=lambda x: x[1], reverse=True)[:top_n]

            # Store results for this term
            for match in top_matches:
                per_term_results.append(
                    {
                        "search_term": term,
                        "matched_text": match[0],
                        "doc_id": match[2],
                        "score": match[1],
                    }
                )

            # Store all scores for combined scoring
            for match in matches:
                doc_id = match[2]
                score = match[1]
                if doc_id not in all_scores:
                    all_scores[doc_id] = []
                all_scores[doc_id].append(score)

    # Convert per-term results to DataFrame
    per_term_df = pd.DataFrame(per_term_results)
    if not per_term_df.empty:
        per_term_df = per_term_df.sort_values(
            ["search_term", "score"], ascending=[True, False]
        )

    # Calculate combined scores
    combined_scores = {}
    for doc_id, scores in all_scores.items():
        # Combined score calculation:
        # - Average of max scores for each term that matched
        # - Weighted by number of terms that matched
        # - Normalized by total number of search terms
        combined_scores[doc_id] = {
            "doc_id": doc_id,
            "matched_text": main_series[doc_id],
            "avg_score": np.mean(scores),
            "max_score": max(scores),
            "matching_terms": len(scores),
            "total_terms": len(search_terms),
            "combined_score": (np.mean(scores) * len(scores)) / len(search_terms),
        }

    # Convert combined scores to DataFrame and sort
    combined_df = pd.DataFrame(combined_scores.values())
    if not combined_df.empty:
        combined_df = combined_df.sort_values("combined_score", ascending=False)

    return per_term_df, combined_df


def get_matched_source(combined_results, question):
    """
    Get document IDs from combined_results if available, otherwise fall back to question source.

    Parameters:
    - combined_results: DataFrame with potential 'doc_id' column
    - question: dict containing 'source' as fallback

    Returns:
    - list of transformed IDs
    """
    if not combined_results.empty and "doc_id" in combined_results.columns:
        return [doc_id.split("_")[0] for doc_id in combined_results["doc_id"]]
    return question["source"]


def finance_main(vector_store, question, doc_set, score_threshold=90) -> List[Document]:
    targets = question["source"]  # get all source list from this question

    main_string = filter_docs(targets, doc_set)  # get docs accroding to source list
    q_keywords = question["parsed_query"]["keyword"]
    # q_keywords.append(question['parsed_query']['season'])
    substring = pd.Series(q_keywords)  # keywords ready to search

    # Get both per-term and combined results
    term_results, combined_results = advanced_fuzzy_search(
        main_string, substring, top_n=3
    )

    if not combined_results.empty:
        results_over_criteria = combined_results[
            combined_results.avg_score >= score_threshold
        ]
        if not results_over_criteria.empty:
            matched_source = get_matched_source(results_over_criteria, question)
        else:
            matched_source = [str(i) for i in question["source"]]
            # matched_source = ["-1"]
            # print(f"qid: {question['qid']} has no avg score over 80 in fuzzy search")
    else:
        matched_source = [str(i) for i in question["source"]]
        # matched_source = ["-1"]
        # print(f"qid: {question['qid']} has no result in fuzzy search")

    # question['query'] = ';'.join(question["parsed_query"]["keyword"])
    question["query"] = question["parsed_query"]["scenario"]
    question["source"] = matched_source

    return dense_search_with_cross_encoder(
        vector_store=vector_store,
        question=question,
        k_dense=5,
        k_cross=1,
    )


def main(finance_question_set, doc_set):
    search_result = {}
    search_list = []
    answer = []
    score_threshold = 90

    for question in tqdm(finance_question_set):
        targets = question["source"]  # get all source list from this question

        main_string = filter_docs(targets, doc_set)  # get docs accroding to source list
        q_keywords = question["parsed_query"]["keyword"]
        # q_keywords.append(question['parsed_query']['season'])
        substring = pd.Series(q_keywords)  # keywords ready to search

        # Get both per-term and combined results
        term_results, combined_results = advanced_fuzzy_search(
            main_string, substring, top_n=3
        )

        if not combined_results.empty:
            results_over_criteria = combined_results[
                combined_results.avg_score >= score_threshold
            ]
            if not results_over_criteria.empty:
                matched_source = get_matched_source(results_over_criteria, question)
            else:
                matched_source = [str(i) for i in question["source"]]
                # matched_source = ["-1"]
                # print(f"qid: {question['qid']} has no avg score over 80 in fuzzy search")
        else:
            matched_source = [str(i) for i in question["source"]]
            # matched_source = ["-1"]
            # print(f"qid: {question['qid']} has no result in fuzzy search")

        search_result.update({question["qid"]: matched_source})
        search_list.append(
            {
                "qid": question["qid"],
                "retrieve": [
                    int(matched_source[i]) for i in range(len(matched_source))
                ],
                "category": question["category"],
            }
        )

        # question['query'] = ';'.join(question["parsed_query"]["keyword"])
        question["query"] = question["parsed_query"]["scenario"]
        question["source"] = matched_source

        query_response = dense_search_with_cross_encoder(
            vector_store=vector_store,
            question=question,
            k_dense=5,
            k_cross=1,
        )

        answer.append(
            {
                "qid": question["qid"],
                "retrieve": int(query_response[0].metadata["source_id"]),
                "category": query_response[0].metadata["category"],
            }
        )

    with open("./brute_80_dense_cross_scenario_v4.json", "w") as a:
        json.dump({"answers": answer}, a, indent=4, ensure_ascii=False)
