
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
        timeout=300,
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
    executed_tools: list[str]
    evaluation: dict
    follow_up_plan: list[dict]
    evaluation_error: str

def planning_node(state: EDAState) -> dict:
    """Ask the LLM to create a validated analysis plan."""
    try:
        plan = create_analysis_plan(state["df"])
        return {"plan": plan, "error": ""}
    except Exception as exc:
        return {"error": f"Planning failed: {exc}"}


def get_tool_functions(df: pd.DataFrame, output_dir: str | Path) -> dict:
    """Return the approved tools the agent is allowed to execute."""
    return {
        "analyze_missing_values": lambda: analyze_missing_values(df),
        "analyze_numeric_columns": lambda: analyze_numeric_columns(df),
        "detect_outliers": lambda: detect_outliers(df),
        "analyze_correlations": lambda: analyze_correlations(df),
        "generate_visualizations": lambda: generate_visualizations(
            df, output_dir=output_dir
        ),
    }


def execute_tools(
    state: EDAState,
    steps: list[dict],
) -> tuple[dict, list[str]]:
    """Execute approved tools and capture their results."""
    df = state["df"]
    output_dir = state["output_dir"]
    tool_functions = get_tool_functions(df, output_dir)

    results = {}
    executed = []

    for step in steps:
        tool_name = step.get("tool")

        # Never execute a tool outside the approved allowlist.
        if tool_name not in ALLOWED_TOOLS:
            continue

        if tool_name not in tool_functions or tool_name in executed:
            continue

        try:
            results[tool_name] = tool_functions[tool_name]()
        except Exception as exc:
            results[tool_name] = {"error": str(exc)}

        # Record attempted tools so failed tools aren't repeated endlessly.
        executed.append(tool_name)

    return results, executed


def execution_node(state: EDAState) -> dict:
    """Execute the initial AI-generated plan."""
    if state.get("error"):
        return {}

    results, executed = execute_tools(
        state,
        state["plan"].get("plan", []),
    )

    return {
        "results": results,
        "executed_tools": executed,
    }


def make_results_json_safe(results: dict) -> dict:
    """Convert analysis results into data that can be sent to Ollama."""
    safe_results = {}

    for tool_name, result in results.items():
        if isinstance(result, pd.DataFrame):
            safe_results[tool_name] = result.to_dict(orient="records")
        elif isinstance(result, list):
            safe_results[tool_name] = [str(item) for item in result]
        elif isinstance(result, dict):
            safe_results[tool_name] = result
        else:
            safe_results[tool_name] = str(result)

    return safe_results


def evaluate_results(results, executed_tools):
    """Evaluate actual EDA results and recommend evidence-based follow-up."""

    available_tools = sorted(ALLOWED_TOOLS)

    evidence = {
        "executed_tools": executed_tools,
        "available_tools": available_tools,
        "results": make_results_json_safe(results),
    }

    system_prompt = """
You are a strict, evidence-based exploratory data analysis evaluator.

Your job is to summarize the supplied computed EDA results and decide
whether additional analysis would be useful.

STRICT RULES:
1. Use only facts supported by the supplied results.
2. Never invent statistics, values, patterns, or conclusions.
3. Do not claim a distribution is normal unless a normality test result
   is explicitly supplied.
4. Do not claim a dataset has no significant outliers merely because
   the supplied results show only a few outliers.
5. Distinguish correlation from causation.
6. If a result is missing or unavailable, say so rather than guessing.
7. Recommend follow-up tools only when they provide meaningful additional
   information.
8. Only recommend tools listed in available_tools.
9. Do not recommend tools that are already listed in executed_tools.
10. Recommend no more than two follow-up tools.
11. If no additional analysis is needed, return an empty follow_up_tools list.
12. Return valid JSON only. Do not use Markdown code fences.

Required JSON format:
{
  "summary": "A concise evidence-based summary of the actual results.",
  "follow_up_tools": [
    {
      "tool": "analyze_correlations",
      "reason": "Explain why this analysis would be useful based on the evidence."
    }
  ]
}

The follow_up_tools array may be empty.
"""

    user_prompt = (
        "Evaluate these actual EDA results. Do not infer findings that are "
        "not supported by the evidence.\n\n"
        + json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
        )[:20000]
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "format": "json",
        },
        timeout=300,
    )

    response.raise_for_status()

    response_data = response.json()
    content = response_data["message"]["content"]

    parsed = json.loads(content)

    # Validate the summary.
    summary = parsed.get("summary", "")

    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(
            "The AI evaluator returned an empty or invalid summary."
        )

    # Validate follow-up recommendations.
    proposed_tools = parsed.get("follow_up_tools", [])

    if not isinstance(proposed_tools, list):
        proposed_tools = []

    validated_follow_ups = []
    already_executed = set(executed_tools)

    for recommendation in proposed_tools:
        if not isinstance(recommendation, dict):
            continue

        tool_name = recommendation.get("tool")
        reason = recommendation.get("reason", "")

        if not isinstance(tool_name, str):
            continue

        if tool_name not in ALLOWED_TOOLS:
            continue

        if tool_name in already_executed:
            continue

        if not isinstance(reason, str) or not reason.strip():
            continue

        validated_follow_ups.append(
            {
                "tool": tool_name,
                "reason": reason.strip(),
            }
        )

        if len(validated_follow_ups) >= 2:
            break

    return {
        "summary": summary.strip(),
        "follow_up_tools": validated_follow_ups,
    }


def evaluation_node(state: EDAState) -> dict:
    """Evaluate initial results and plan one optional follow-up round."""
    try:
        evaluation = evaluate_results(
            state.get("results", {}),
            state.get("executed_tools", []),
        )

        return {
            "evaluation": evaluation,
            "follow_up_plan": evaluation["follow_up_tools"],
            "evaluation_error": "",
        }

    except Exception as exc:
        # A failed evaluation should not discard successful EDA results.
        return {
            "evaluation": {
                "summary": "Evaluation could not be completed.",
                "follow_up_tools": [],
            },
            "follow_up_plan": [],
            "evaluation_error": str(exc),
        }


def followup_execution_node(state: EDAState) -> dict:
    """Execute the evaluator's validated, one-time follow-up plan."""
    results, executed = execute_tools(
        state,
        state.get("follow_up_plan", []),
    )

    combined_results = dict(state.get("results", {}))
    combined_results.update(results)

    combined_executed = list(state.get("executed_tools", []))
    combined_executed.extend(executed)

    return {
        "results": combined_results,
        "executed_tools": combined_executed,
    }


def build_eda_graph():
    """Build a bounded plan, execute, evaluate workflow."""
    workflow = StateGraph(EDAState)

    workflow.add_node("plan", planning_node)
    workflow.add_node("execute", execution_node)
    workflow.add_node("evaluate", evaluation_node)
    workflow.add_node("followup", followup_execution_node)

    workflow.add_edge(START, "plan")

    workflow.add_conditional_edges(
        "plan",
        lambda state: "stop" if state.get("error") else "continue",
        {
            "stop": END,
            "continue": "execute",
        },
    )

    workflow.add_edge("execute", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        lambda state: (
            "followup"
            if state.get("follow_up_plan")
            else "stop"
        ),
        {
            "followup": "followup",
            "stop": END,
        },
    )

    # No edge returns to the evaluator: follow-up is limited to one round.
    workflow.add_edge("followup", END)

    return workflow.compile()


def run_agent(
    dataset_path: str | Path,
    output_dir: str | Path | None = None,
):
    """Run the EDA agent on a dataset and save charts to a chosen folder."""

    dataset_path = Path(dataset_path).resolve()

    project_root = Path(__file__).resolve().parent.parent

    if output_dir is None:
        output_dir = project_root / "reports" / "charts"
    else:
        output_dir = Path(output_dir)

        if not output_dir.is_absolute():
            output_dir = project_root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)

    initial_state = {
        "df": df,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "plan": [],
        "results": {},
        "error": None,
        "executed_tools": [],
        "evaluation": {},
        "follow_up_plan": [],
        "evaluation_error": None,
    }

    graph = build_eda_graph()
    final_state = graph.invoke(initial_state)

    return final_state

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
                    print(result.to_string())
                elif isinstance(result, list):
                    for item in result:
                        print(item)
                else:
                    print(json.dumps(result, indent=2, default=str))

            print("\n========== AI EVALUATION ==========")

            evaluation = final_state.get("evaluation", {})
            print(
                evaluation.get(
                    "summary",
                    "No evaluation available.",
                )
            )

            follow_up_tools = evaluation.get("follow_up_tools", [])

            if follow_up_tools:
                print("\nAdditional tools recommended:")
                for step in follow_up_tools:
                    print(f"- {step['tool']}: {step['reason']}")
            else:
                print("\nNo additional analysis was recommended.")

            if final_state.get("evaluation_error"):
                print(
                    "\nEvaluation warning:",
                    final_state["evaluation_error"],
                )

            print(
                "\nTools attempted:",
                ", ".join(final_state.get("executed_tools", [])),
            )

            print("\nEDA agent workflow completed.")

    except Exception as exc:
        print(f"\nThe EDA workflow failed: {exc}")
