
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel dataset and validate the file."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Use CSV, XLSX, or XLS."
        )

    if df.empty:
        raise ValueError("The uploaded dataset contains no rows.")

    if len(df.columns) == 0:
        raise ValueError("The dataset contains no columns.")

    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """Summarize the dataset's structure and data types."""

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_bytes": int(
            df.memory_usage(deep=True).sum()
        ),
    }


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Count missing values and calculate their percentages."""

    missing_count = df.isna().sum()

    result = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": (
            missing_count / len(df) * 100
        ).round(2),
    })

    return result.sort_values(
        "missing_count", ascending=False
    )


def analyze_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics for numeric columns."""

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.describe().T.round(2)


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag potential outliers using the IQR method."""

    numeric_df = df.select_dtypes(include="number")
    results = []

    for column in numeric_df.columns:
        values = numeric_df[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (values < lower_bound)
            | (values > upper_bound)
        )

        results.append({
            "column": str(column),
            "lower_bound": round(float(lower_bound), 2),
            "upper_bound": round(float(upper_bound), 2),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_values": values[outlier_mask].tolist(),
        })

    return pd.DataFrame(results)


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlations between numeric columns."""

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr().round(3)


def run_basic_eda(df: pd.DataFrame) -> dict:
    """Run the basic EDA analyses and collect their results."""

    return {
        "profile": profile_dataset(df),
        "missing_values": analyze_missing_values(df),
        "numeric_statistics": analyze_numeric_columns(df),
        "outliers": detect_outliers(df),
        "correlations": analyze_correlations(df),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "customers.csv"

    dataset = load_dataset(dataset_path)
    results = run_basic_eda(dataset)

    print("\n========== DATASET PROFILE ==========")
    print(results["profile"])

    print("\n========== MISSING VALUES ==========")
    print(results["missing_values"])

    print("\n========== NUMERIC STATISTICS ==========")
    print(results["numeric_statistics"])

    print("\n========== POTENTIAL OUTLIERS ==========")
    print(results["outliers"].to_string(index=False))

    print("\n========== CORRELATIONS ==========")
    print(results["correlations"])
