import argparse
import json


def calculate_precision_at_1(ground_truths_path, predictions_path):
    # Load ground truths and predictions
    with open(ground_truths_path, 'r') as f:
        ground_truths = json.load(f)['ground_truths']

    with open(predictions_path, 'r') as f:
        predictions = json.load(f)['answers']

    # Convert lists to dictionaries for easier lookup
    ground_truth_dict = {item['qid']: item['retrieve'] for item in ground_truths}
    predicted_dict = {item['qid']: item['retrieve'] for item in predictions}

    # Calculate Precision@1
    correct = 0
    total = len(ground_truth_dict)

    for qid, correct_answer in ground_truth_dict.items():
        if qid in predicted_dict and predicted_dict[qid] == correct_answer:
            correct += 1

    precision_at_1 = correct / total

    print(f'Precision@1: {precision_at_1:.7f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--ground_truths_path', type=str, required=True)
    parser.add_argument('-p', '--predictions_path', type=str, required=True)
    args = parser.parse_args()

    calculate_precision_at_1(args.ground_truths_path, args.predictions_path)
