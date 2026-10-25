
import json
import os
from pathlib import Path

import pandas as pd
import requests

from app.eda_engine import load_dataset, profile_dataset


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/chat",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

ALLOWED_TOOLS = {
    "analyze_missing_values",
    "analyze_numeric_columns",
    "detect_outliers",
    "analyze_correlations",
    "generate_visualizations",
}


def build_dataset_summary(df: pd.DataFrame) -> dict:
    """Build a compact summary for the LLM without sending every row."""

    profile = profile_dataset(df)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    return {
        "profile": profile,
        "missing_values": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
        },
        "numeric_columns": numeric_columns,
        "non_numeric_columns": df.select_dtypes(
            exclude="number"
        ).columns.tolist(),
        "numeric_statistics": (
            df[numeric_columns]
            .describe()
            .round(2)
            .to_dict()
            if numeric_columns
            else {}
        ),
    }


def ask_ollama(summary: dict, available_tools: list[str]) -> dict:
    """Ask the local LLM to propose an evidence-based EDA plan."""

    system_prompt = """
You are an experienced data analyst planning exploratory data analysis.

Choose useful analyses based only on the supplied dataset summary.
Do not claim you have executed any analysis.
Do not invent findings.
Return a JSON object with this structure:

{
  "plan": [
    {
      "tool": "analyze_missing_values",
      "reason": "Explain briefly why this analysis is useful."
    }
  ],
  "overall_goal": "Briefly explain the main analysis priorities."
}

Only choose tools listed in available_tools.
Return no more than five plan items.
Do not include Markdown or text outside the JSON object.
"""

    user_payload = {
        "dataset_summary": summary,
        "available_tools": available_tools,
    }

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, default=str),
                },
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=180,
    )

    response.raise_for_status()

    message = response.json()["message"]["content"]
    result = json.loads(message)

    if not isinstance(result, dict):
        raise ValueError("The model response must be a JSON object.")

    return result


def validate_plan(
    proposed_result: dict,
    available_tools: list[str],
) -> dict:
    """Reject unknown tools, malformed steps, and duplicate actions."""

    raw_plan = proposed_result.get("plan", [])

    if not isinstance(raw_plan, list):
        raise ValueError("The model's plan must be a list.")

    allowed = set(available_tools)
    validated_plan = []
    seen_tools = set()

    for item in raw_plan:
        if not isinstance(item, dict):
            continue

        tool = item.get("tool")
        reason = item.get("reason", "")

        if not isinstance(tool, str) or tool not in allowed:
            continue

        if tool in seen_tools:
            continue

        if not isinstance(reason, str):
            reason = ""

        validated_plan.append({
            "tool": tool,
            "reason": reason[:500],
        })

        seen_tools.add(tool)

        if len(validated_plan) >= 5:
            break

    if not validated_plan:
        raise ValueError(
            "The model did not return any valid analysis tools. "
            "Try again or inspect the model response."
        )

    goal = proposed_result.get("overall_goal", "")

    if not isinstance(goal, str):
        goal = ""

    return {
        "overall_goal": goal[:1000],
        "plan": validated_plan,
    }


def create_analysis_plan(df: pd.DataFrame) -> dict:
    """Generate and validate an analysis plan for a dataset."""

    summary = build_dataset_summary(df)

    available_tools = sorted(ALLOWED_TOOLS)

    proposed_result = ask_ollama(
        summary=summary,
        available_tools=available_tools,
    )

    return validate_plan(
        proposed_result=proposed_result,
        available_tools=available_tools,
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "customers.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Place the sample dataset at: {dataset_path}"
        )

    dataset = load_dataset(dataset_path)
    print("Dataset loaded successfully.")
    print(f"Connecting to Ollama model: {OLLAMA_MODEL}")

    try:
        analysis_plan = create_analysis_plan(dataset)

        print("\n========== AI ANALYSIS PLAN ==========")
        print(json.dumps(analysis_plan, indent=2))

    except requests.RequestException as exc:
        print("\nCould not connect to Ollama.")
        print("Check that Ollama is running and the model is installed.")
        print(f"Details: {exc}")

    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print("\nCould not create a valid analysis plan.")
        print(f"Details: {exc}")
