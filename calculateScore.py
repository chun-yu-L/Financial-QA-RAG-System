import argparse
import json


def calculate_precision_at_1(ground_truths_path, predictions_path, output_file):
    # Load ground truths and predictions
    with open(ground_truths_path, "r") as f:
        ground_truths = json.load(f)["ground_truths"]

    with open(predictions_path, "r") as f:
        predictions = json.load(f)["answers"]

    # Convert lists to dictionaries for easier lookup
    ground_truth_dict = {item["qid"]: item for item in ground_truths}
    predicted_dict = {item["qid"]: item["retrieve"] for item in predictions}

    # Organize results by category
    categories = {}
    total_correct = 0
    total_questions = 0
    wrong_predictions_overall = []

    for qid, ground_truth in ground_truth_dict.items():
        category = ground_truth["category"]
        if category not in categories:
            categories[category] = {"correct": 0, "total": 0, "wrong_predictions": []}
        categories[category]["total"] += 1
        total_questions += 1

        if qid in predicted_dict:
            predicted_answer = predicted_dict[qid]
            if predicted_answer == ground_truth["retrieve"]:
                categories[category]["correct"] += 1
                total_correct += 1
            else:
                # Store incorrect predictions
                wrong_pred = {
                    "qid": qid,
                    "ground_truth": ground_truth["retrieve"],
                    "prediction": predicted_answer,
                    "category": category,
                }
                categories[category]["wrong_predictions"].append(wrong_pred)
                wrong_predictions_overall.append(wrong_pred)

    # Calculate total Precision@1
    total_precision_at_1 = total_correct / total_questions
    print(f"Total Precision@1: {total_precision_at_1:.7f}")

    # Prepare output content
    output_content = f"Total Precision@1: {total_precision_at_1:.7f}\n\n"

    # Write category-wise precision and wrong predictions
    for category, results in categories.items():
        precision_at_1 = results["correct"] / results["total"]
        output_content += f"Category: {category}, Precision@1: {precision_at_1:.7f}\n"
        output_content += "Wrong Predictions:\n"
        for wrong_pred in results["wrong_predictions"]:
            output_content += f"QID: {wrong_pred['qid']}, Ground Truth: {wrong_pred['ground_truth']}, Prediction: {wrong_pred['prediction']}\n"
        output_content += "\n"

    # Write to output file
    with open(output_file, "w") as output_f:
        output_f.write(output_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--ground_truths_path", type=str, required=True)
    parser.add_argument("-p", "--predictions_path", type=str, required=True)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    args = parser.parse_args()

    calculate_precision_at_1(
        args.ground_truths_path, args.predictions_path, args.output_file
    )
