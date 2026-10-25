# Agentic EDA Pipeline

An AI-powered Exploratory Data Analysis (EDA) system that combines traditional data analysis, local large language models, and an interactive dashboard to automate dataset exploration and reporting.

The system uses **Ollama, LangGraph, Pandas, Plotly, and Streamlit** to plan analysis tasks, execute analytical tools, evaluate findings, and generate visualizations and HTML reports.

## Features

- **Automated dataset profiling** — inspect dataset dimensions, column types, memory usage, and duplicate rows.
- **Data quality analysis** — identify missing values and duplicate records.
- **Statistical analysis** — calculate descriptive statistics for numeric features.
- **Outlier detection** — identify potential outliers using the Interquartile Range (IQR) method.
- **Correlation analysis** — examine relationships between numeric variables.
- **AI-driven planning** — use a local LLM to create an analysis plan.
- **Agentic execution** — execute approved analytical tools through a LangGraph workflow.
- **AI result evaluation** — evaluate findings and suggest limited follow-up analysis.
- **Interactive visualizations** — generate Plotly histograms, box plots, and correlation heatmaps.
- **Streamlit dashboard** — upload datasets, preview data, and explore analysis results.
- **HTML report generation** — create a downloadable report containing analysis results and chart links.
- **Report bundle download** — package the HTML report and generated charts into a ZIP archive.
- **Local LLM support** — run AI planning and evaluation using Ollama.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Visualization | Plotly |
| AI model runtime | Ollama |
| Default LLM | `qwen2.5:7b` |
| Agent workflow | LangGraph |
| Dashboard | Streamlit |
| Statistical analysis | SciPy, scikit-learn |
| Excel support | openpyxl |

## Project Structure

```text
Agentic-EDA-Pipeline/
├── app/
│   ├── __init__.py
│   ├── eda_engine.py
│   ├── visualizations.py
│   ├── agent.py
│   └── report_generator.py
├── data/
│   └── customers.csv
├── reports/
│   └── runs/
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

- `eda_engine.py` — dataset profiling, missing-value analysis, statistics, outlier detection, and correlations.
- `visualizations.py` — generates interactive HTML charts.
- `agent.py` — defines the AI planning, tool execution, evaluation, and follow-up workflow.
- `report_generator.py` — generates HTML analysis reports.
- `streamlit_app.py` — provides the interactive dashboard.
- `data/` — contains sample datasets.
- `reports/runs/` — stores outputs from individual analysis runs.

## Prerequisites

Install the following before running the project:

- Python 3.11 recommended
- Git
- Ollama
- The `qwen2.5:7b` model, or another compatible model configured for the project

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ali241124/Agentic-EDA-Pipeline.git
cd Agentic-EDA-Pipeline
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Python 3.11 is not available, use an installed Python version compatible with the dependencies.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from [ollama.com](https://ollama.com/).

Download the default model:

```powershell
ollama pull qwen2.5:7b
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

If Ollama is already running in the background, you do not need to start it again.

Verify the model is available:

```powershell
ollama list
```

### 5. Configure environment variables (optional)

The default configuration uses the local Ollama API and the `qwen2.5:7b` model. If your code supports environment-based configuration, create a `.env` file or set these variables in PowerShell:

```text
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=qwen2.5:7b
```

Do not commit API keys, credentials, or other secrets.

## Running the Application

From the project root, activate your virtual environment and start Streamlit:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Open the local URL printed in your terminal, usually:

`http://localhost:8501`

## How to Use

1. Open the Streamlit dashboard.
2. Upload a CSV or supported Excel dataset.
3. Review the dataset preview.
4. Start the EDA workflow.
5. Explore the profiling, data quality, statistical, outlier, and correlation results.
6. Review the AI-generated analysis plan and evaluation.
7. Explore the generated interactive charts.
8. Download the HTML report and chart bundle.
9. Extract the ZIP archive before opening the HTML report so its relative chart links can resolve correctly.

## Analysis Workflow

```text
Dataset Upload
      |
      v
Dataset Profiling
      |
      v
AI Analysis Planning
      |
      v
Analytical Tool Execution
      |
      v
Result Evaluation
      |
      v
Optional Follow-up Analysis
      |
      v
Charts and HTML Report
      |
      v
Streamlit Dashboard and Downloads
```

## Analytical Methods

### Missing Values

Reports missing-value counts and percentages for each column.

### Numeric Statistics

Calculates descriptive statistics for numeric measurement columns. Identifier columns such as `customer_id` are excluded from measurement analysis.

### Outlier Detection

Uses the IQR method:

- Lower bound = Q1 − 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR

Values outside these bounds are flagged as potential outliers, not automatically classified as errors.

### Correlation Analysis

Calculates correlations between eligible numeric measurement columns. Correlation indicates association and does not establish causation.

### Visualizations

The visualization module can generate histograms, box plots, and a correlation heatmap for eligible numeric features.

## Example Dataset

The repository includes `data/customers.csv` as a sample dataset for testing the pipeline.

You can replace it with your own CSV or Excel dataset through the dashboard.

Use appropriately sized datasets for your machine and local LLM configuration.

## Limitations

- AI-generated plans and summaries may be incomplete or incorrect and should be checked against the underlying analytical results.
- Analysis speed depends on dataset size, system resources, and the local LLM.
- IQR-based outlier detection identifies statistical anomalies; domain knowledge is needed to determine whether they are genuine errors.
- Correlation does not imply causation.
- This project is a development and portfolio project; production deployment requires additional security, validation, testing, and resource management.
- HTML report chart links rely on the report and chart files retaining their relative directory structure.

## Future Improvements

- Automated unit and integration tests
- More robust handling of malformed and very large datasets
- Additional statistical and data quality checks
- Improved AI evaluation and evidence-grounded summaries
- More export formats and self-contained report options
- Deployment with suitable resource limits and security controls
- Logging, monitoring, and performance optimization

## Contributing

Contributions, suggestions, and bug reports are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request describing the improvement.

## Author

**Syed Ali Hassan**

- GitHub: [Ali241124](https://github.com/Ali241124)
- Portfolio: [syedalihassan-portfolio.vercel.app](https://syedalihassan-portfolio.vercel.app/)

## License

No license has been specified yet. Add a `LICENSE` file if you intend to publish the project under an open-source license.
