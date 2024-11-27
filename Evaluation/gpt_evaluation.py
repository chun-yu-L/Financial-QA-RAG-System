import os
import re
import json
import argparse
import logging

from dotenv import load_dotenv

from tqdm.auto import tqdm

from utils import get_prompt, REFEREE_Model


def load_dataset(predict_dataset_path, ground_truths_path, query_dataset_path):
    '''
    Read json files and return queries, ground truths, and predictions.
    '''
    with open(predict_dataset_path, 'r', encoding='utf-8') as f:
        predict_data = json.load(f)
    with open(ground_truths_path, 'r', encoding='utf-8') as f:
        ground_truths_data = json.load(f)
    with open(query_dataset_path, 'r', encoding='utf-8') as f:
        query_data = json.load(f)

    queries = [data['query'] for data in query_data['questions']]
    ground_truths = [data['generate'] for data in ground_truths_data['ground_truths']]
    predictions = [data['generate'] for data in predict_data['answers']]

    return queries, ground_truths, predictions


def parse_response(resp: str, idx: int):
    """
    Parse the response from the model.
    建議必要時增加錯誤處理，確保自行測試模型時的穩定性。
    """

    logging.basicConfig(level=logging.INFO, filename='log_parse_response.txt',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S',
	)

    parsed_resp = re.sub(r'^```json\s*|```$', '', resp.strip())
    resp = parsed_resp.lower()
    answer = -1

    try:
        model_resp = json.loads(resp)
        logging.info(f"idx: {idx}\tresponse: {resp}")

        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True) or (
                isinstance(model_resp["accuracy"], str) and
                model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
    except Exception as e:
        logging.error(f"error: {e} \n idx={idx}\nresponse: {resp}")

    return answer


def evaluate_predictions(queries: list, ground_truths: list, predictions: list):
    '''
    queries: list of queries
    ground_truths: list of ground truths
    predictions: list of predictions
    '''
    referee_model = REFEREE_Model(
        azure_deployment_name=azure_deployment_name,
        azure_openai_api_key=azure_openai_api_key, 
        api_version=api_version, 
        azure_endpoint=azure_endpoint
    )

    n_miss, n_correct = 0, 0

    # system_prompt
    INSTRUCTIONS, IN_CONTEXT_EXAMPLES = get_prompt()
    system_prompt = f"{INSTRUCTIONS}\n{IN_CONTEXT_EXAMPLES}\n"

    for _idx, prediction in enumerate(tqdm(predictions, total=len(predictions), desc="Evaluating Predictions")):
        query = queries[_idx]
        ground_truth = ground_truths[_idx].strip()

        # 當模型預測 "不知道" 時 (字串需完全相符)，將其視為 "missing"
        if "不知道" == prediction:
            n_miss += 1
            continue
        # 當模型預測完全相同，可直接視為 "correct"
        elif prediction == ground_truth:
            n_correct += 1
            continue
        messages = referee_model.create_prompt(system_prompt, query, ground_truth, prediction)
        response = referee_model.generate_response(messages, _idx+1)
        if response:
            eval_res = parse_response(response, _idx+1)
            if eval_res == 1:
                n_correct += 1

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1
    }

    with open("score.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return results


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="GPT Evaluation.")
    parser.add_argument('--predict_dataset_path', type=str, required=True, help='Path to the prediction json file.')
    parser.add_argument('--ground_truths_path', type=str, required=True, help='Path to the ground truths json file.')
    parser.add_argument('--query_dataset_path', type=str, required=True, help='Path to the query dataset.')
    args = parser.parse_args()

    azure_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
    api_version=os.getenv("AZURE_OPENAI_API_VERSION") 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")

    # Load dataset
    queries, ground_truths, predictions = load_dataset(args.predict_dataset_path, args.ground_truths_path, args.query_dataset_path)

    # Evaluate predictions
    results = evaluate_predictions(queries, ground_truths, predictions)
    print(results)
