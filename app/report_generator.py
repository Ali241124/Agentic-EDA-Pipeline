
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

from app.eda_engine import profile_dataset


def _safe_text(value):
    """Convert a value to HTML-safe text."""
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass

    return escape(str(value))


def _render_table(data, title):
    """Render a DataFrame or dictionary as an HTML table."""
    heading = f"<h2>{escape(title)}</h2>"

    if isinstance(data, pd.DataFrame):
        if data.empty:
            return heading + "<p>No results available.</p>"

        return heading + data.to_html(
            index=True,
            border=0,
            classes="data-table",
            na_rep="N/A",
            escape=True,
        )

    if isinstance(data, dict):
        if not data:
            return heading + "<p>No results available.</p>"

        rows = []

        for key, value in data.items():
            rows.append(
                f"<tr><th>{escape(str(key))}</th>"
                f"<td>{_safe_text(value)}</td></tr>"
            )

        return (
            heading
            + '<table class="data-table"><tbody>'
            + "".join(rows)
            + "</tbody></table>"
        )

    return heading + f"<pre>{_safe_text(data)}</pre>"


def _render_outliers(outliers):
    """Render IQR outlier counts and values."""
    if not isinstance(outliers, dict) or not outliers:
        return "<h2>Outlier Analysis</h2><p>No results available.</p>"

    rows = []

    for column, details in outliers.items():
        if not isinstance(details, dict):
            continue

        count = details.get("outlier_count", 0)
        values = details.get("outlier_values", [])

        rows.append(
            f"<tr><td>{escape(str(column))}</td>"
            f"<td>{_safe_text(count)}</td>"
            f"<td>{_safe_text(values)}</td></tr>"
        )

    return (
        "<h2>Outlier Analysis</h2>"
        '<table class="data-table">'
        "<thead><tr><th>Column</th><th>Count</th>"
        "<th>Values</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def generate_html_report(
    df: pd.DataFrame,
    results: dict,
    output_dir: str | Path,
    dataset_name: str = "Dataset",
    plan=None,
    evaluation=None,
    executed_tools=None,
) -> Path:
    """Create an HTML report inside the current analysis run folder."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = results if isinstance(results, dict) else {}

    # Support both the basic EDA result names and agent tool names.
    profile = results.get("profile", profile_dataset(df))

    missing = results.get(
        "missing_values",
        results.get("analyze_missing_values", pd.DataFrame()),
    )

    numeric = results.get(
        "numeric_statistics",
        results.get("analyze_numeric_columns", pd.DataFrame()),
    )

    outliers = results.get(
        "outliers",
        results.get("detect_outliers", {}),
    )

    correlations = results.get(
        "correlations",
        results.get("analyze_correlations", pd.DataFrame()),
    )

    if isinstance(evaluation, dict):
        summary = evaluation.get("summary", "")
        follow_ups = evaluation.get("follow_up_tools", [])
    elif isinstance(evaluation, str):
        summary = evaluation
        follow_ups = []
    else:
        summary = ""
        follow_ups = []

    plan = plan if isinstance(plan, list) else []
    executed_tools = (
        executed_tools if isinstance(executed_tools, list) else []
    )

    def render_list(items):
        if not items:
            return "<li>None available.</li>"

        return "".join(
            f"<li>{escape(str(item))}</li>"
            for item in items
        )

    followup_items = []

    for item in follow_ups:
        if isinstance(item, dict):
            tool = escape(str(item.get("tool", "Unknown")))
            reason = escape(str(item.get("reason", "")))
            followup_items.append(
                f"<li><strong>{tool}</strong>: {reason}</li>"
            )

    if not followup_items:
        followup_items = ["<li>No additional recommendations.</li>"]

    chart_files = sorted(
        path for path in output_dir.glob("*.html")
        if path.name != "eda_report.html"
    )

    if chart_files:
        chart_links = "".join(
            f'<li><a href="{escape(path.name)}">'
            f'{escape(path.stem.replace("_", " ").title())}</a></li>'
            for path in chart_files
        )
    else:
        chart_links = "<li>No charts found in this run folder.</li>"

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    generated_at = datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EDA Report - {escape(dataset_name)}</title>
<style>
* {{ box-sizing: border-box; }}
body {{
    margin: 0;
    background: #f3f5f9;
    color: #192337;
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.6;
}}
header {{
    background: #172033;
    color: white;
    padding: 32px max(24px, calc((100% - 1050px) / 2));
}}
header p {{ color: #cbd5e1; }}
main {{
    max-width: 1100px;
    margin: auto;
    padding: 24px;
}}
section {{
    background: white;
    border: 1px solid #dce2ea;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
    overflow-x: auto;
}}
h2 {{ margin-top: 0; font-size: 20px; }}
.metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 12px;
}}
.metric {{
    padding: 16px;
    border: 1px solid #dce2ea;
    border-radius: 9px;
}}
.metric span {{ display: block; color: #647084; }}
.metric strong {{ font-size: 26px; }}
.data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}}
.data-table th, .data-table td {{
    padding: 10px;
    border-bottom: 1px solid #e1e6ee;
    text-align: left;
    vertical-align: top;
}}
.data-table th {{ background: #f1f4f8; }}
a {{ color: #3157a4; }}
.muted {{ color: #647084; font-size: 13px; }}
footer {{ padding: 12px 0; color: #647084; }}
</style>
</head>
<body>
<header>
    <h1>Exploratory Data Analysis Report</h1>
    <p>Dataset: {escape(dataset_name)}</p>
    <p>Generated: {escape(generated_at)}</p>
</header>
<main>
    <section>
        <h2>Dataset Overview</h2>
        <div class="metrics">
            <div class="metric">
                <span>Rows</span>
                <strong>{len(df)}</strong>
            </div>
            <div class="metric">
                <span>Columns</span>
                <strong>{len(df.columns)}</strong>
            </div>
            <div class="metric">
                <span>Missing cells</span>
                <strong>{missing_cells}</strong>
            </div>
            <div class="metric">
                <span>Duplicate rows</span>
                <strong>{duplicate_rows}</strong>
            </div>
        </div>
        <p class="muted">
            IQR outliers are potential anomalies, not necessarily errors.
            Correlation indicates association, not causation.
        </p>
    </section>

    <section>
        {_render_table(profile, "Dataset Profile")}
    </section>

    <section>
        {_render_table(missing, "Missing-Value Analysis")}
    </section>

    <section>
        {_render_table(numeric, "Numeric Statistics")}
    </section>

    <section>
        {_render_outliers(outliers)}
    </section>

    <section>
        {_render_table(correlations, "Correlation Matrix")}
    </section>

    <section>
        <h2>AI Evaluation</h2>
        <p>{escape(summary) if summary else "No AI summary available."}</p>
        <h3>Analysis Plan</h3>
        <ul>{render_list(plan)}</ul>
        <h3>Tools Executed</h3>
        <ul>{render_list(executed_tools)}</ul>
        <h3>Follow-up Recommendations</h3>
        <ul>{"".join(followup_items)}</ul>
    </section>

    <section>
        <h2>Generated Charts</h2>
        <ul>{chart_links}</ul>
        <p class="muted">
            Keep this report in the same folder as its chart files
            so the links continue to work.
        </p>
    </section>

    <footer>Generated by Agentic EDA Pipeline.</footer>
</main>
</body>
</html>
"""

    report_path = output_dir / "eda_report.html"
    report_path.write_text(html, encoding="utf-8")

    return report_path
