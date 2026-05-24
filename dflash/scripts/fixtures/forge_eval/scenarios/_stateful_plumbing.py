"""Stateful plumbing scenarios — basic_2step, sequential_3step, error_recovery."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .._forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario, _check, _placeholder_workflow


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


# ── Backend 1: CountryFactsDB ──────────────────────────────────


class CountryFactsDB:
    def __init__(self) -> None:
        self.data = {
            "france": "Capital: Paris. Population: 2.1 million (city), 67 million (country).",
            "japan": "Capital: Tokyo. Population: 14 million (city), 125 million (country).",
        }
        self.last_retrieved: str | None = None

    def get_country_info(self, country: str) -> str:
        key = country.strip().lower()
        if key in self.data:
            self.last_retrieved = self.data[key]
            return self.data[key]
        return f"No entry found for '{country}'."

    def summarize(self, content: str) -> str:
        return content  # echo-back terminal


def _build_basic_2step_stateful() -> tuple[Workflow, callable]:
    db = CountryFactsDB()
    tools: dict[str, ToolDef] = {
        "get_country_info": ToolDef(
            spec=ToolSpec(
                name="get_country_info",
                description="Look up facts about a country.",
                parameters=CountryParams,
            ),
            callable=lambda **kw: db.get_country_info(kw["country"]),
        ),
        "summarize": ToolDef(
            spec=ToolSpec(
                name="summarize",
                description="Summarize content and provide the final answer.",
                parameters=ContentParams,
            ),
            callable=lambda **kw: db.summarize(kw.get("content", "")),
        ),
    }
    workflow = Workflow(
        name="basic_2step_stateful",
        description="Look up country facts, then summarize",
        tools=tools,
        required_steps=["get_country_info"],
        terminal_tool="summarize",
        system_prompt_template=(
            "You are a helpful assistant. Use the available tools to answer "
            "the user's question. First use get_country_info to retrieve "
            "information, then use summarize to provide the final answer."
        ),
    )
    validate_state = lambda: db.last_retrieved is not None
    return workflow, validate_state


basic_2step_stateful = EvalScenario(
    name="basic_2step_stateful",
    description="Stateful 2-step — country lookup with argument-dependent results.",
    workflow=_placeholder_workflow("basic_2step_stateful", "summarize", ["get_country_info"]),
    user_message="What is the capital of France?",
    validate=lambda args: _check(args.get("content", ""), ["paris", "capital"]),
    build_workflow=_build_basic_2step_stateful,
    tags=["stateful", "plumbing"],
    ideal_iterations=2,
)


# ── Backend 2: SalesPipeline ───────────────────────────────────


class SalesPipeline:
    def __init__(self) -> None:
        self.datasets: dict[tuple[int, int], dict[str, Any]] = {
            (4, 2024): {
                "records": 150, "columns": 12,
                "summary": "Dataset: 150 records, 12 columns, covering Q1–Q4 2024 sales data.",
                "analysis": "Analysis: Revenue grew 23% YoY. Top product: Widget Pro. Weakest region: APAC.",
            },
        }
        self.loaded_data: dict[str, Any] | None = None
        self.analysis: str | None = None

    def fetch_sales_data(self, quarter: int, year: int) -> str:
        key = (quarter, year)
        if key in self.datasets:
            self.loaded_data = self.datasets[key]
            return self.loaded_data["summary"]
        return f"No data found for Q{quarter} {year}."

    def analyze_sales(self) -> str:
        if self.loaded_data is None:
            return "Error: no data loaded. Call fetch_sales_data first."
        self.analysis = self.loaded_data["analysis"]
        return self.analysis

    def report(self, findings: str) -> str:
        return findings  # echo-back terminal


def _build_sequential_3step_stateful() -> tuple[Workflow, callable]:
    db = SalesPipeline()
    tools: dict[str, ToolDef] = {
        "fetch_sales_data": ToolDef(
            spec=ToolSpec(
                name="fetch_sales_data",
                description="Fetch sales data for a given quarter and year.",
                parameters=FetchSalesParams,
            ),
            callable=lambda **kw: db.fetch_sales_data(kw["quarter"], kw["year"]),
        ),
        "analyze_sales": ToolDef(
            spec=ToolSpec(
                name="analyze_sales",
                description="Analyze the loaded sales data and produce findings.",
                parameters=EmptyParams,
            ),
            callable=lambda **kw: db.analyze_sales(),
        ),
        "report": ToolDef(
            spec=ToolSpec(
                name="report",
                description="Produce a final report from findings.",
                parameters=FindingsParams,
            ),
            callable=lambda **kw: db.report(kw.get("findings", "")),
        ),
    }
    workflow = Workflow(
        name="sequential_3step_stateful",
        description="Fetch sales data, analyze, then report",
        tools=tools,
        required_steps=["fetch_sales_data", "analyze_sales"],
        terminal_tool="report",
        system_prompt_template=(
            "You are a data analyst assistant. Fetch the sales data first, "
            "then analyze it, then produce a report using the report tool."
        ),
    )
    validate_state = lambda: (
        db.loaded_data is not None
        and db.analysis is not None
    )
    return workflow, validate_state


sequential_3step_stateful = EvalScenario(
    name="sequential_3step_stateful",
    description="Stateful 3-step — sales pipeline with argument-dependent data loading.",
    workflow=_placeholder_workflow("sequential_3step_stateful", "report", ["fetch_sales_data", "analyze_sales"]),
    user_message="Generate a sales report from the Q4 2024 dataset.",
    validate=lambda args: _check(args.get("findings", ""), ["23", "widget pro", "apac"]),
    build_workflow=_build_sequential_3step_stateful,
    tags=["stateful", "plumbing"],
    ideal_iterations=3,
)


# ── Backend 3: RecordFetcher ──────────────────────────────────


class RecordFetcher:
    """Stateful backend matching lambda error_recovery — same tool shape."""

    def __init__(self) -> None:
        self.fetched_count: int | None = None

    def fetch(self, count: str) -> str:
        if not (isinstance(count, str) and len(count) == 4 and count.isdigit()):
            raise TypeError(
                f"count must be a zero-padded 4-digit string, got '{count}'"
            )
        self.fetched_count = int(count)
        return f"Fetched {self.fetched_count} records."

    def summarize(self, content: str) -> str:
        return content  # echo-back terminal


def _build_error_recovery_stateful() -> tuple[Workflow, callable]:
    db = RecordFetcher()
    tools: dict[str, ToolDef] = {
        "fetch": ToolDef(
            spec=ToolSpec(
                name="fetch",
                description="Fetch records. The count parameter must be a numeric string.",
                parameters=CountParams,
            ),
            callable=lambda **kw: db.fetch(kw["count"]),
        ),
        "summarize": ToolDef(
            spec=ToolSpec(
                name="summarize",
                description="Summarize the fetched content.",
                parameters=ContentParams,
            ),
            callable=lambda **kw: db.summarize(kw.get("content", "")),
        ),
    }
    workflow = Workflow(
        name="error_recovery_stateful",
        description="Fetch with validation, then summarize",
        tools=tools,
        required_steps=["fetch"],
        terminal_tool="summarize",
        system_prompt_template=(
            "You are a helpful assistant. Fetch the requested records, "
            "then summarize them."
        ),
    )
    validate_state = lambda: db.fetched_count == 10
    return workflow, validate_state


error_recovery_stateful = EvalScenario(
    name="error_recovery_stateful",
    description="Stateful error recovery — format trip with state tracking.",
    workflow=_placeholder_workflow("error_recovery_stateful", "summarize", ["fetch"]),
    user_message="Fetch 10 records and summarize them.",
    validate=lambda args: _check(args.get("content", ""), ["10", "record"]),
    build_workflow=_build_error_recovery_stateful,
    tags=["stateful", "plumbing"],
    ideal_iterations=3,
)
