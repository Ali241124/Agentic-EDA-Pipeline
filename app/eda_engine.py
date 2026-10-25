
from pathlib import Path

import pandas as pd


# Common identifier columns that should not be treated as measurements.
IDENTIFIER_COLUMNS = {
    "id",
    "customer_id",
    "user_id",
    "record_id",
    "row_id",
}


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel dataset and validate its contents."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Please use CSV, XLSX, or XLS."
        )

    if df.empty:
        raise ValueError(
            "The uploaded dataset is empty."
        )

    if len(df.columns) == 0:
        raise ValueError(
            "The dataset contains no columns."
        )

    # Remove accidental whitespace from column names.
    df.columns = df.columns.astype(str).str.strip()

    return df


def get_analysis_numeric_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Return numeric measurement columns, excluding identifiers."""

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    analysis_columns = []

    for column in numeric_columns:
        normalized_name = column.strip().lower()

        if normalized_name in IDENTIFIER_COLUMNS:
            continue

        if normalized_name.endswith("_id"):
            continue

        analysis_columns.append(column)

    return analysis_columns


def profile_dataset(df: pd.DataFrame) -> dict:
    """Summarize the dataset's structure and basic properties."""

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "memory_usage_bytes": int(
            df.memory_usage(deep=True).sum()
        ),
    }


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing-value counts and percentages."""

    missing_count = df.isnull().sum()

    missing_percentage = (
        missing_count / len(df) * 100
    ).round(2)

    results = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
    })

    # Show columns with missing values first.
    results = results.sort_values(
        by="missing_count",
        ascending=False,
    )

    return results


def analyze_numeric_columns(df: pd.DataFrame):
    """Calculate descriptive statistics for numeric measurements."""

    numeric_columns = get_analysis_numeric_columns(df)

    if not numeric_columns:
        return pd.DataFrame()

    return df[numeric_columns].describe().T.round(2)


def detect_outliers(df: pd.DataFrame) -> dict:
    """Detect numeric outliers using the interquartile range (IQR)."""

    numeric_columns = get_analysis_numeric_columns(df)
    results = {}

    for column in numeric_columns:
        values = df[column].dropna()

        if values.empty:
            results[column] = {
                "outlier_count": 0,
                "outlier_values": [],
            }
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = values[
            (values < lower_bound)
            | (values > upper_bound)
        ]

        results[column] = {
            "outlier_count": int(len(outliers)),
            "outlier_values": outliers.tolist(),
        }

    return results


def analyze_correlations(df: pd.DataFrame):
    """Calculate correlations between numeric measurements."""

    numeric_columns = get_analysis_numeric_columns(df)

    if len(numeric_columns) < 2:
        return pd.DataFrame()

    return df[numeric_columns].corr().round(3)


def run_basic_eda(df: pd.DataFrame) -> dict:
    """Run the core exploratory data analysis functions."""

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

    print("\n" + "=" * 60)
    print("DATASET PROFILE")
    print("=" * 60)
    print(results["profile"])

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(results["missing_values"])

    print("\n" + "=" * 60)
    print("NUMERIC STATISTICS")
    print("=" * 60)
    print(results["numeric_statistics"])

    print("\n" + "=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)
    print(results["outliers"])

    print("\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)
    print(results["correlations"])
    print("\nBasic EDA completed successfully.")
