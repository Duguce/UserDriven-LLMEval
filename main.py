from src.data_preprocessing import preprocess_data
from src.metrics_calculation import compute_metrics
from src.scoring import calculate_final_score, optimize_weights


def main():
    # Step 1: Preprocess the raw data
    print("Step 1: Preprocessing data...")
    processed_data = preprocess_data("data/raw_data.xlsx")
    processed_data.to_csv("data/processed_data.csv", index=False)
    print("Processed data saved to data/processed_data.csv")

    # Step 2: Compute composite metrics
    print("Step 2: Computing metrics...")
    metrics_data = compute_metrics(processed_data)
    metrics_data.to_csv("data/metrics_data.csv", index=False)
    print("Metrics data saved to data/metrics_data.csv")

    # Step 3: Optimize weights and calculate final scores
    print("Step 3: Optimizing weights and calculating final scores...")
    best_weights = optimize_weights(metrics_data)
    print("Optimal weights:", best_weights)

    metrics_data["Final Score"] = calculate_final_score(metrics_data, best_weights)
    metrics_data.to_csv("data/final_scores.csv", index=False)
    print("Final scores saved to data/final_scores.csv")


if __name__ == "__main__":
    main()
