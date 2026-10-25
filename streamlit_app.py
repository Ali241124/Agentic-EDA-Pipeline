import io
import zipfile
from pathlib import Path
import os
import tempfile
import uuid
import pandas as pd
import streamlit as st
from app.report_generator import generate_html_report
from app.agent import run_agent
from app.eda_engine import profile_dataset


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Agentic EDA | Data Intelligence",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Agentic EDA Pipeline")
st.caption(
    "AI-assisted exploratory data analysis powered by "
    "LangGraph, Ollama, Pandas, and Plotly."
)

st.divider()


# --------------------------------------------------
# Sidebar: dataset upload and controls
# --------------------------------------------------

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV and Excel.",
    )

    st.caption(
        "Your dataset is processed by this Python application. "
        "The AI planner uses your locally running Ollama model."
    )

    run_button = st.button(
        "Run AI-powered EDA",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    st.divider()
    st.subheader("Workflow")
    st.markdown(
        """
        1. Load and preview the dataset
        2. Generate an AI analysis plan
        3. Execute approved EDA tools
        4. Evaluate findings
        5. Run optional follow-up analysis
        """
    )


# --------------------------------------------------
# Initial state
# --------------------------------------------------

if "eda_state" not in st.session_state:
    st.session_state["eda_state"] = None

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None


# --------------------------------------------------
# Preview uploaded dataset
# --------------------------------------------------

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()

        if uploaded_file.name.lower().endswith(".csv"):
            preview_df = pd.read_csv(
                pd.io.common.BytesIO(file_bytes)
            )
        else:
            preview_df = pd.read_excel(
                pd.io.common.BytesIO(file_bytes)
            )

        if preview_df.empty:
            st.warning("The uploaded dataset contains no rows.")
        else:
            st.subheader("Dataset Preview")
            st.write(f"File: **{uploaded_file.name}**")

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Rows", f"{preview_df.shape[0]:,}")
            metric2.metric("Columns", f"{preview_df.shape[1]:,}")
            metric3.metric(
                "Missing cells",
                f"{int(preview_df.isna().sum().sum()):,}",
            )

            st.dataframe(
                preview_df.head(10),
                use_container_width=True,
            )

    except Exception as exc:
        st.error(f"Could not preview this dataset: {exc}")


# --------------------------------------------------
# Run the LangGraph EDA agent
# --------------------------------------------------

if run_button and uploaded_file is not None:
    temporary_path = None

    try:
        suffix = Path(uploaded_file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(uploaded_file.getvalue())
            temporary_path = temporary_file.name

        with st.spinner(
            "The AI agent is planning and running the analysis. "
            "This may take a few minutes..."
        ):
            run_id = uuid.uuid4().hex[:10]

            run_output_dir = (
                Path(__file__).resolve().parent
                / "reports"
                / "runs"
                / run_id
            )

            final_state = run_agent(
                temporary_path,
                output_dir=run_output_dir,
            )
        
        report_path = generate_html_report(
            df=final_state["df"],
            results=final_state.get("results", {}),
            output_dir=run_output_dir,
            dataset_name=uploaded_file.name,
            plan=final_state.get("plan", []),
            evaluation=final_state.get("evaluation", {}),
            executed_tools=final_state.get("executed_tools", []),
        )


        st.session_state["eda_report_path"] = str(report_path)
        st.session_state["eda_run_output_dir"] = str(run_output_dir)

        if final_state.get("error"):
            st.error(final_state["error"])
        else:
            st.session_state["eda_state"] = final_state
            st.session_state["dataset_name"] = uploaded_file.name
            st.success("EDA workflow completed successfully.")

    except Exception as exc:
        st.error("The EDA workflow failed.")
        st.exception(exc)

    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


# --------------------------------------------------
# Results dashboard
# --------------------------------------------------

final_state = st.session_state.get("eda_state")

if final_state:
    df = final_state["df"]
    plan = final_state.get("plan", {})
    results = final_state.get("results", {})
    evaluation = final_state.get("evaluation", {})

    st.divider()
    st.header("Analysis Dashboard")
    st.caption(
        f"Dataset: {st.session_state.get('dataset_name', 'Uploaded dataset')}"
    )

    tab_overview, tab_quality, tab_charts, tab_ai = st.tabs(
        [
            "Overview",
            "Data Quality",
            "Visualizations",
            "AI Analysis",
        ]
    )
    
report_path = st.session_state.get("eda_report_path")
run_output_dir = st.session_state.get("eda_run_output_dir")

if (
    report_path
    and Path(report_path).is_file()
    and run_output_dir
    and Path(run_output_dir).is_dir()
):
    zip_buffer = io.BytesIO()
    run_dir = Path(run_output_dir)

    # Bundle the report and all generated charts together.
    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:
        for file_path in run_dir.rglob("*"):
            if file_path.is_file():
                zip_file.write(
                    file_path,
                    arcname=file_path.relative_to(run_dir),
                )

    st.download_button(
        label="Download Report + Charts (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="eda_report_bundle.zip",
        mime="application/zip",
        key="download_eda_report_bundle",
    )

    # ----------------------------------------------
    # Overview
    # ----------------------------------------------

    with tab_overview:
        profile = profile_dataset(df)

        st.subheader("Dataset Profile")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", f"{len(df.columns):,}")
        col3.metric(
            "Duplicate rows",
            f"{int(df.duplicated().sum()):,}",
        )
        col4.metric(
            "Missing cells",
            f"{int(df.isna().sum().sum()):,}",
        )

        with st.expander("Column names and data types"):
            dtype_df = pd.DataFrame(
                {
                    "Column": df.columns.astype(str),
                    "Data type": [
                        str(dtype) for dtype in df.dtypes
                    ],
                    "Missing values": [
                        int(count)
                        for count in df.isna().sum()
                    ],
                }
            )
            st.dataframe(dtype_df, use_container_width=True)

        st.subheader("Full Dataset")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download dataset as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="analyzed_dataset.csv",
            mime="text/csv",
        )

    # ----------------------------------------------
    # Data quality
    # ----------------------------------------------

    with tab_quality:
        st.subheader("Missing Values")

        missing_result = results.get("analyze_missing_values")

        if isinstance(missing_result, pd.DataFrame):
            st.dataframe(
                missing_result,
                use_container_width=True,
            )
        else:
            st.info("Missing-value analysis was not executed.")

        st.subheader("Outlier Detection")

        outlier_result = results.get("detect_outliers")

        if isinstance(outlier_result, pd.DataFrame):
            st.dataframe(
                outlier_result,
                use_container_width=True,
            )
        else:
            st.info("Outlier detection was not executed.")

        st.subheader("Numeric Statistics")

        numeric_result = results.get("analyze_numeric_columns")

        if isinstance(numeric_result, pd.DataFrame):
            st.dataframe(
                numeric_result,
                use_container_width=True,
            )
        else:
            st.info("Numeric analysis was not executed.")

    # ----------------------------------------------
    # Interactive charts
    # ----------------------------------------------

    with tab_charts:
        st.subheader("Generated Visualizations")

        chart_paths = results.get("generate_visualizations", [])

        available_charts = [
            Path(chart_path)
            for chart_path in chart_paths
            if Path(chart_path).is_file()
            and Path(chart_path).suffix.lower() == ".html"
        ]

        if available_charts:
            selected_chart = st.selectbox(
                "Choose a chart",
                available_charts,
                format_func=lambda path: path.stem.replace(
                    "_", " "
                ).title(),
            )

            chart_html = selected_chart.read_text(
                encoding="utf-8"
            )

            st.components.v1.html(
                chart_html,
                height=620,
                scrolling=True,
            )

            st.download_button(
                "Download selected chart",
                data=chart_html,
                file_name=selected_chart.name,
                mime="text/html",
            )

            with st.expander("Show all generated chart files"):
                for chart_path in available_charts:
                    st.write(chart_path.name)

        else:
            st.info(
                "No charts are available. Run the EDA workflow "
                "and include the visualization tool in its plan."
            )

        correlation_result = results.get("analyze_correlations")

        if isinstance(correlation_result, pd.DataFrame):
            st.subheader("Correlation Matrix")
            st.dataframe(
                correlation_result,
                use_container_width=True,
            )

    # ----------------------------------------------
    # AI analysis and evaluation
    # ----------------------------------------------

    with tab_ai:
        st.subheader("AI Analysis Plan")
        st.write(
            plan.get(
                "overall_goal",
                "No overall analysis goal was returned.",
            )
        )

        for index, step in enumerate(
            plan.get("plan", []),
            start=1,
        ):
            st.markdown(
                f"**{index}. {step['tool']}**"
            )
            st.write(step.get("reason", ""))

        st.divider()
        st.subheader("AI Evaluation")

        evaluation_summary = evaluation.get("summary", "")

        if evaluation_summary:
            st.write(evaluation_summary)
        else:
            st.warning(
                "The AI evaluation is unavailable. "
                "Check the agent output for an evaluation error."
            )

        follow_up_tools = evaluation.get("follow_up_tools", [])

        if follow_up_tools:
            st.markdown("**Recommended follow-up analyses**")

            for step in follow_up_tools:
                st.markdown(f"- **{step['tool']}**: {step['reason']}")
        else:
            st.write("No additional analyses were recommended.")

        if final_state.get("evaluation_error"):
            st.warning(
                "Evaluation warning: "
                + final_state["evaluation_error"]
            )

        st.subheader("Tools Attempted")
        st.write(
            ", ".join(final_state.get("executed_tools", []))
            or "Tool execution details are unavailable."
        )

else:
    st.info(
        "Upload a CSV or Excel dataset using the sidebar, "
        "preview it, and select **Run AI-powered EDA** to begin."
    )
