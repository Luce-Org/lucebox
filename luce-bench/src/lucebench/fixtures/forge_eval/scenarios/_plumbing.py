"""Plumbing scenarios — basic FC, sequential steps, error recovery."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .._forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario, _check

# ── Pydantic parameter models ──────────────────────────────────


class CountryParams(BaseModel):
    country: str = Field(description="Country name")


class ContentParams(BaseModel):
    content: str = Field(description="The content to summarize")


class FetchSalesParams(BaseModel):
    quarter: int = Field(description="Quarter number (1-4)")
    year: int = Field(description="Four-digit year")


class EmptyParams(BaseModel):
    pass


class FindingsParams(BaseModel):
    findings: str = Field(description="The findings to include in the report")


class CountParams(BaseModel):
    count: str = Field(description="Number of records to fetch (must be a numeric string)")


# ── Scenario 1: basic_2step ──────────────────────────────────────

_basic_2step_tools: dict[str, ToolDef] = {
    "get_country_info": ToolDef(
        spec=ToolSpec(
            name="get_country_info",
            description="Look up facts about a country.",
            parameters=CountryParams,
        ),
        callable=lambda **kwargs: "The capital of France is Paris. Population: 2.1 million.",
    ),
    "summarize": ToolDef(
        spec=ToolSpec(
            name="summarize",
            description="Summarize content and provide the final answer.",
            parameters=ContentParams,
        ),
        callable=lambda **kwargs: kwargs.get("content", ""),
    ),
}

basic_2step = EvalScenario(
    name="basic_2step",
    description="Baseline FC check — does the model do function calling at all?",
    workflow=Workflow(
        name="basic_2step",
        description="Simple 2-step information retrieval and summary",
        tools=_basic_2step_tools,
        required_steps=["get_country_info"],
        terminal_tool="summarize",
        system_prompt_template=(
            "You are a helpful assistant. Use the available tools to answer "
            "the user's question. First use get_country_info to retrieve "
            "information, then use summarize to provide the final answer."
        ),
    ),
    user_message="What is the capital of France?",
    validate=lambda args: _check(args.get("content", ""), ["paris", "capital"]),
    tags=["plumbing"],
)


# ── Scenario 2: sequential_3step ────────────────────────────────

_sequential_3step_tools: dict[str, ToolDef] = {
    "fetch_sales_data": ToolDef(
        spec=ToolSpec(
            name="fetch_sales_data",
            description="Fetch sales data for a given quarter and year.",
            parameters=FetchSalesParams,
        ),
        callable=lambda **kwargs: "Dataset: 150 records, 12 columns, covering Q1–Q4 2024 sales data.",
    ),
    "analyze_sales": ToolDef(
        spec=ToolSpec(
            name="analyze_sales",
            description="Analyze the loaded sales data and produce findings.",
            parameters=EmptyParams,
        ),
        callable=lambda **kwargs: "Analysis: Revenue grew 23% YoY. Top product: Widget Pro. Weakest region: APAC.",
    ),
    "report": ToolDef(
        spec=ToolSpec(
            name="report",
            description="Produce a final report from findings.",
            parameters=FindingsParams,
        ),
        callable=lambda **kwargs: kwargs.get("findings", ""),
    ),
}

sequential_3step = EvalScenario(
    name="sequential_3step",
    description="Required step enforcement — 3-step sequential workflow.",
    workflow=Workflow(
        name="sequential_3step",
        description="Fetch data, analyze, then report",
        tools=_sequential_3step_tools,
        required_steps=["fetch_sales_data", "analyze_sales"],
        terminal_tool="report",
        system_prompt_template=(
            "You are a data analyst assistant. Fetch the sales data first, "
            "then analyze it, then produce a report using the report tool."
        ),
    ),
    user_message="Generate a sales report from the Q4 2024 dataset.",
    validate=lambda args: _check(args.get("findings", ""), ["23", "widget pro", "apac"]),
    tags=["plumbing"],
)


# ── Scenario 3: error_recovery ───────────────────────────────────


def _fetch_with_validation(**kwargs: Any) -> str:
    count = kwargs.get("count", "")
    if not (isinstance(count, str) and len(count) == 4 and count.isdigit()):
        raise TypeError(
            f"count must be a zero-padded 4-digit string, got '{count}'"
        )
    return f"Fetched {int(count)} records."


_error_recovery_tools: dict[str, ToolDef] = {
    "fetch": ToolDef(
        spec=ToolSpec(
            name="fetch",
            description="Fetch records. The count parameter must be a numeric string.",
            parameters=CountParams,
        ),
        callable=_fetch_with_validation,
    ),
    "summarize": ToolDef(
        spec=ToolSpec(
            name="summarize",
            description="Summarize the fetched content.",
            parameters=ContentParams,
        ),
        callable=lambda **kwargs: kwargs.get("content", ""),
    ),
}

error_recovery = EvalScenario(
    name="error_recovery",
    description="Tool error self-correction — model must recover from TypeError.",
    workflow=Workflow(
        name="error_recovery",
        description="Fetch with validation, then summarize",
        tools=_error_recovery_tools,
        required_steps=["fetch"],
        terminal_tool="summarize",
        system_prompt_template=(
            "You are a helpful assistant. Fetch the requested records, "
            "then summarize them."
        ),
    ),
    user_message="Fetch 10 records and summarize them.",
    validate=lambda args: _check(args.get("content", ""), ["10", "record"]),
    tags=["plumbing"],
)
