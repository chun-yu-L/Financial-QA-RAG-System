import argparse
import gc
import json
import logging
import os
from datetime import datetime

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from graph.graph_builder import build_workflow
from graph.state import QAState

logging.basicConfig(
    level=logging.INFO,
    filename="log.txt",
    format="[%(asctime)s %(levelname)-8s] %(message)s",
    datefmt="%Y%m%d %H:%M:%S",
)


def main(question_set, doc_set):
    """
    Main function to generate answers.

    This function takes a question set and a document set, construct a workflow graph,
    and invokes the graph to generate answers for each question.

    The results are stored in a JSON file named "generation_result.json"
    and "pred.json".

    Args:
        question_set (Dict[str, Any]): A dictionary containing question data.
            The dictionary should contain a key "questions" whose value is a list
            of dictionaries, each containing a question. Each question dictionary
            should contain at least two keys: "qid" and "query".
        doc_set (Dict[str, Any]): A dictionary containing document data.
            The dictionary should contain a key "docs" whose value is a list of
            dictionaries, each containing a document. Each document dictionary
            should contain at least one key: "id".

    Returns:
        None
    """
    start_time = datetime.now()

    load_dotenv()
    client = QdrantClient(url=os.getenv("qdrant_url"), timeout=30)

    results = []

    for question in question_set["questions"]:
        # 構建 workflow
        app = build_workflow()
        initial_state = QAState(
            question=question,
            doc_set=doc_set,
            retrieve_doc=None,
            client=client,
            answer={},
        )

        result_state = app.invoke(initial_state)
        results.append(result_state["answer"])

        del app
        gc.collect()
        torch.cuda.empty_cache()

    with open("./generation_result.json", "w") as Output:
        json.dump({"answers": results}, Output, ensure_ascii=False, indent=4)

    output = {
        "answers": [
            {"qid": answer["qid"], "generate": answer["generate"]} for answer in results
        ]
    }

    with open("pred.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

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
