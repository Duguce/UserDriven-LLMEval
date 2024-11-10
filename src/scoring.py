from typing import Tuple

import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

from .metrics_calculation import compute_metrics


def objective(weights: Tuple[float, float], data: pd.DataFrame) -> float:
    """Define the objective function for optimization to maximize the average score distance."""
    w1, w2 = weights

    # Increase the weight of Hugging Face metrics
    scores = (1.5 * w1) * data["UEI"] * data["TWF"] + w2 * data["CRR"]

    # Convert scores to a numpy array for integer-based indexing
    scores = scores.values

    # Calculate the average distance
    n = len(scores)
    avg_distance = sum(
        abs(scores[i] - scores[j]) for i in range(n) for j in range(i + 1, n)
    ) / (n * (n - 1))

    # Add a regularization term to penalize if w1 and w2 are too close
    regularization = abs(w1 - w2) * 0.2
    return -(avg_distance - regularization)  # Negative to perform minimization


def optimize_weights(data: pd.DataFrame) -> Tuple[float, float]:
    """Automatically determine optimal weights using Bayesian optimization."""
    # Limit the search range of w1 to focus on Hugging Face-related metrics
    space = [Real(1, 3, name="w1"), Real(0, 1, name="w2")]
    result = gp_minimize(lambda w: objective(w, data), space, n_calls=100, x0=[2, 0.5])
    return result.x  # Return the optimal weights


def calculate_final_score(
    data: pd.DataFrame, weights: Tuple[float, float]
) -> pd.Series:
    """Calculate the final score based on optimal weights."""
    w1, w2 = weights
    return w1 * data["UEI"] * data["TWF"] + w2 * data["CRR"]


if __name__ == "__main__":
    # Load preprocessed data
    data = pd.read_csv("data/metrics_data.csv")

    # Check if there are NaN or inf values in the data
    if data.isnull().values.any():
        print("Error: Data contains NaN values. Please check the input data.")
        print(data.isnull().sum())
    elif (data == float("inf")).values.any() or (data == float("-inf")).values.any():
        print("Error: Data contains inf values. Please check the input data.")
    else:
        # Compute composite metrics
        data = compute_metrics(data)

        # Perform optimization to get the best weights
        best_weights = optimize_weights(data)
        print("Optimal weights:", best_weights)

        # Calculate the final score based on the optimal weights
        data["Final Score"] = calculate_final_score(data, best_weights)

        # Save the final score results
        data.to_csv("data/final_scores.csv", index=False)
        print("Final scores saved to data/final_scores.csv")
