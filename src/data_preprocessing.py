from datetime import datetime
from typing import List, Union

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from an Excel file."""
    return pd.read_excel(file_path)


def cal_months_since_release(
    data: pd.DataFrame, ref_date: Union[str, datetime] = "2024-11-09"
) -> pd.DataFrame:
    """Calculate the number of months from the model's release date to the specified reference date."""
    if isinstance(ref_date, str):
        ref_date = datetime.strptime(ref_date, "%Y-%m-%d")
    data["Release Date"] = pd.to_datetime(data["Release Date"], format="%Y年%m月%d日")
    data["Months Since Release"] = data["Release Date"].apply(
        lambda date: (ref_date.year - date.year) * 12 + (ref_date.month - date.month)
    )
    return data


def remove_outliers(
    data: pd.DataFrame, columns: List[str], quantile: float = 0.99
) -> pd.DataFrame:
    """Removes outliers in the specified columns, defaulting to values above the 99th percentile."""
    for column in columns:
        threshold = data[column].quantile(quantile)
        data = data[data[column] <= threshold]
    return data


def min_max_normalize(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Performs Min-Max normalization for the specified columns."""
    for column in columns:
        min_val = data[column].min()
        max_val = data[column].max()
        data[column] = (data[column] - min_val) / (max_val - min_val)
    return data


def preprocess_data(file_path: str) -> pd.DataFrame:
    """Complete preprocessing process, including release month calculation, outlier removal, and normalization."""
    data = load_data(file_path)

    data = cal_months_since_release(data)

    columns_to_remove_outliers = [
        "Monthly Downloads (HF)",
        "Total Likes (HF)",
        "Total Stars (GitHub)",
    ]
    data = remove_outliers(data, columns_to_remove_outliers)

    columns_to_normalize = [
        "Monthly Downloads (HF)",
        "Total Likes (HF)",
        "Total Stars (GitHub)",
        "Open Issues (GitHub)",
        "Closed Issues (GitHub)",
        "Open PRs (GitHub)",
        "Closed PRs (GitHub)",
    ]
    data = min_max_normalize(data, columns_to_normalize)

    return data


if __name__ == "__main__":
    data = preprocess_data("data/raw_data.xlsx")
    data.to_csv("data/processed_data.csv", index=False)
