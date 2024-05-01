import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def print_usage():
    print(
        """Usage:
    python3 NADI2024-ST2-Scorer.py NADI2024_subtask2_dev1_GOLD.txt UBC_subtask2_dev_1.txt
        """
    )


# Load the file with predicitons
def load_scores(filename):
    with open(filename) as f:
        predictions_scores = [float(l.strip()) for l in f]

    return predictions_scores


def compute_RMSE(gold, predicted):
    """
    Compute the Root Mean Squared Error (RMSE) between the gold and predicted values.

    Args:
    gold: list of gold values in [0, 1]
    predicted: list of predicted values

    Returns:
    RMSE: the Root Mean Squared Error
    """

    assert len(gold) == len(
        predicted
    ), "gold and predicted lists are not the same length"
    sum_square_errors = sum([(gold[i] - predicted[i]) ** 2 for i in range(len(gold))])

    return (sum_square_errors / len(gold)) ** 0.5


if __name__ == "__main__":
    verbose = 0
    if len(sys.argv) > 4 or len(sys.argv) < 3:
        print_usage()
        exit()

    if len(sys.argv) == 4 and sys.argv[3] != "-verbose":
        print_usage()
        exit()

    if len(sys.argv) == 4:
        verbose = 1

    gold_file = sys.argv[1]
    pred_file = sys.argv[2]

    gold_scores = load_scores(gold_file)
    predicted_scores = load_scores(pred_file)

    if len(gold_scores) != len(predicted_scores):
        print("both files must have same number of instances")
        exit()

    RMSE = compute_RMSE(gold_scores, predicted_scores)

    print("\nRMSE: %.5f" % RMSE)

    # write to a text file
    with open(pred_file.split("/")[-1].split(".")[0] + "_result.txt", "w") as out_file:
        out_file.write("\nRMSE: %.5f" % RMSE)
