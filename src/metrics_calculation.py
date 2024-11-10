import pandas as pd


def calculate_uei(data: pd.DataFrame) -> pd.Series:
    """Calculate User Engagement Index (UEI)."""
    return (
        data["Total Likes (HF)"] / data["Months Since Release"]
        + data["Total Stars (GitHub)"] / data["Months Since Release"]
        + data["Monthly Downloads (HF)"]
    )


def calculate_crr(data: pd.DataFrame) -> pd.Series:
    """Calculate Community Response Rate (CRR)."""
    # If Open Issues + Closed Issues is 0, set CRR to 0
    return data.apply(
        lambda row: (
            row["Closed Issues (GitHub)"]
            / (row["Open Issues (GitHub)"] + row["Closed Issues (GitHub)"])
            if (row["Open Issues (GitHub)"] + row["Closed Issues (GitHub)"]) != 0
            else 0
        ),
        axis=1,
    )


def calculate_twf(
    data: pd.DataFrame, t_ref: int = 12, epsilon: float = 1e-6
) -> pd.Series:
    """Calculate Time Weighting Factor (TWF)."""
    return t_ref / (data["Months Since Release"] + epsilon)


def compute_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all composite metrics and add them to the DataFrame."""
    data["UEI"] = calculate_uei(data)
    data["CRR"] = calculate_crr(data)
    data["TWF"] = calculate_twf(data)
    return data


if __name__ == "__main__":
    data = pd.read_csv("data/processed_data.csv")
    data = compute_metrics(data)
    data.to_csv("data/metrics_data.csv", index=False)
