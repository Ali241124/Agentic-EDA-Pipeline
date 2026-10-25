
from pathlib import Path

import pandas as pd
import plotly.express as px


# Columns that identify records rather than measure a property.
DEFAULT_IDENTIFIER_COLUMNS = {"customer_id", "id"}


def get_numeric_columns(
    df: pd.DataFrame,
    identifier_columns: set[str] | None = None,
) -> list[str]:
    """Select numeric measurement columns, excluding identifiers."""

    identifiers = {
        column.lower()
        for column in (
            identifier_columns or DEFAULT_IDENTIFIER_COLUMNS
        )
    }

    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column.lower() not in identifiers
    ]


def generate_visualizations(
    df: pd.DataFrame,
    output_dir: str | Path = "reports/charts",
) -> list[Path]:
    """Generate interactive charts and save them as HTML files."""

    if df.empty:
        raise ValueError("Cannot visualize an empty dataset.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_columns = get_numeric_columns(df)
    saved_files = []

    # 1. Histograms: inspect each numeric variable's distribution.
    for column in numeric_columns:
        plot_data = df[[column]].dropna()

        if plot_data.empty:
            continue

        fig = px.histogram(
            plot_data,
            x=column,
            nbins=20,
            title=f"Distribution of {column}",
            labels={column: column},
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title=column,
            yaxis_title="Number of records",
        )

        output_path = output_dir / f"{column}_histogram.html"
        fig.write_html(str(output_path))
        saved_files.append(output_path)

    # 2. Box plots: inspect the spread and potential outliers.
    for column in numeric_columns:
        plot_data = df[[column]].dropna()

        if plot_data.empty:
            continue

        fig = px.box(
            plot_data,
            y=column,
            points="outliers",
            title=f"Box Plot of {column}",
        )

        fig.update_layout(template="plotly_white")

        output_path = output_dir / f"{column}_boxplot.html"
        fig.write_html(str(output_path))
        saved_files.append(output_path)

    # 3. Correlation heatmap: compare numeric measurements.
    if len(numeric_columns) >= 2:
        correlation_matrix = df[numeric_columns].corr()

        fig = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Numerical Feature Correlations",
            aspect="auto",
        )

        fig.update_layout(template="plotly_white")

        output_path = output_dir / "correlation_heatmap.html"
        fig.write_html(str(output_path))
        saved_files.append(output_path)

    return saved_files


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "customers.csv"
    output_dir = project_root / "reports" / "charts"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Place your sample dataset at: {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    chart_files = generate_visualizations(df, output_dir)

    print("\n========== GENERATED CHARTS ==========")

    for chart_file in chart_files:
        print(chart_file.resolve())

    print(f"\nSuccessfully generated {len(chart_files)} charts.")
