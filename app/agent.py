
import json
import os
from pathlib import Path

import pandas as pd
import requests
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from app.eda_engine import (
    load_dataset,
    profile_dataset,
    analyze_missing_values,
    analyze_numeric_columns,
    detect_outliers,
    analyze_correlations,
)
from app.visualizations import generate_visualizations


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


class EDAState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    df: pd.DataFrame
    dataset_path: str
    output_dir: str
    plan: dict
    results: dict
    error: str


def planning_node(state: EDAState) -> dict:
    """Ask the LLM to create a validated analysis plan."""

    try:
        plan = create_analysis_plan(state["df"])
        return {"plan": plan, "error": ""}
    except Exception as exc:
        return {"error": f"Planning failed: {exc}"}


def execution_node(state: EDAState) -> dict:
    """Execute only the analysis tools approved in the plan."""

    if state.get("error"):
        return {}

    df = state["df"]
    output_dir = Path(state["output_dir"])

    tool_functions = {
        "analyze_missing_values": lambda: analyze_missing_values(df),
        "analyze_numeric_columns": lambda: analyze_numeric_columns(df),
        "detect_outliers": lambda: detect_outliers(df),
        "analyze_correlations": lambda: analyze_correlations(df),
        "generate_visualizations": lambda: generate_visualizations(
            df, output_dir=output_dir
        ),
    }

    results = {}

    for step in state["plan"]["plan"]:
        tool_name = step["tool"]

        # Defense in depth: never execute an unknown tool.
        if tool_name not in ALLOWED_TOOLS:
            continue

        if tool_name not in tool_functions:
            continue

        try:
            results[tool_name] = tool_functions[tool_name]()
        except Exception as exc:
            # Record the failure and continue with other tools.
            results[tool_name] = {
                "error": str(exc)
            }

    return {"results": results}


def build_eda_graph():
    """Build the initial plan-and-execute LangGraph workflow."""

    workflow = StateGraph(EDAState)

    workflow.add_node("plan", planning_node)
    workflow.add_node("execute", execution_node)

    workflow.add_edge(START, "plan")

    # Skip execution if planning failed.
    workflow.add_conditional_edges(
        "plan",
        lambda state: (
            "stop" if state.get("error") else "continue"
        ),
        {
            "stop": END,
            "continue": "execute",
        },
    )

    workflow.add_edge("execute", END)

    return workflow.compile()


def run_agent(dataset_path: str | Path) -> dict:
    """Run the complete EDA agent workflow on a dataset."""

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = Path(dataset_path).resolve()

    df = load_dataset(dataset_path)

    graph = build_eda_graph()

    initial_state: EDAState = {
        "df": df,
        "dataset_path": str(dataset_path),
        "output_dir": str(project_root / "reports" / "charts"),
        "plan": {},
        "results": {},
        "error": "",
    }

    return graph.invoke(initial_state)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "customers.csv"

    print("Loading dataset and starting the EDA agent...")

    try:
        final_state = run_agent(dataset_path)

        if final_state.get("error"):
            print("\n========== AGENT ERROR ==========")
            print(final_state["error"])
        else:
            print("\n========== AI ANALYSIS PLAN ==========")
            print(json.dumps(final_state["plan"], indent=2))

            print("\n========== EXECUTION RESULTS ==========")

            for tool_name, result in final_state.get(
                "results", {}
            ).items():
                print(f"\n--- {tool_name} ---")

                if isinstance(result, pd.DataFrame):
                    print(result.to_string(index=False))
                elif isinstance(result, list):
                    for item in result:
                        print(item)
                else:
                    print(json.dumps(result, indent=2, default=str))

            print("\nEDA agent workflow completed.")

    except Exception as exc:
        print(f"\nThe EDA workflow failed: {exc}")
