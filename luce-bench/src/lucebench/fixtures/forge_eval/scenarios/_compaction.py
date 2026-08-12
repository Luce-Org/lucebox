"""Compaction-related parameter models and relevance detection scenario."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .._forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario


class CityParams(BaseModel):
    city: str = Field(description="The city name")


class FlightParams(BaseModel):
    origin: str = Field(description="Departure city")
    destination: str = Field(description="Arrival city")


class HotelParams(BaseModel):
    city: str = Field(description="The city to check")
    checkin: str = Field(description="Check-in date")


class CurrencyParams(BaseModel):
    amount: str = Field(description="The amount to convert")
    from_currency: str = Field(description="Source currency code")
    to_currency: str = Field(description="Target currency code")


class ReasonParams(BaseModel):
    reason: str = Field(description="Brief explanation of why no tool is appropriate")


# ── Scenario: relevance_detection ─────────────────────────────

_relevance_detection_tools: dict[str, ToolDef] = {
    "get_forecast": ToolDef(
        spec=ToolSpec(name="get_forecast", description="Get weather forecast for a city.",
                      parameters=CityParams),
        callable=lambda **kwargs: f"Forecast for {kwargs.get('city', '???')}: 22°C, partly cloudy.",
    ),
    "book_flight": ToolDef(
        spec=ToolSpec(name="book_flight", description="Book a flight between two cities.",
                      parameters=FlightParams),
        callable=lambda **kwargs: f"Flight booked: {kwargs.get('origin', '')} → {kwargs.get('destination', '')}.",
    ),
    "check_hotel": ToolDef(
        spec=ToolSpec(name="check_hotel", description="Check hotel availability in a city.",
                      parameters=HotelParams),
        callable=lambda **kwargs: f"3 hotels available in {kwargs.get('city', '')}.",
    ),
    "convert_currency": ToolDef(
        spec=ToolSpec(name="convert_currency", description="Convert an amount between currencies.",
                      parameters=CurrencyParams),
        callable=lambda **kwargs: f"{kwargs.get('amount', '0')} {kwargs.get('from_currency', '')} = 0.00 {kwargs.get('to_currency', '')}.",
    ),
    "decline": ToolDef(
        spec=ToolSpec(name="decline", description="Call this when none of the available tools are relevant to the user's request.",
                      parameters=ReasonParams),
        callable=lambda **kwargs: kwargs.get("reason", ""),
    ),
}

relevance_detection = EvalScenario(
    name="relevance_detection",
    description="Hallucination resistance — model should refuse to call irrelevant tools.",
    workflow=Workflow(
        name="relevance_detection",
        description="Travel tools that are irrelevant to the user's question",
        tools=_relevance_detection_tools,
        required_steps=[],
        terminal_tool="decline",
        system_prompt_template=(
            "You are a helpful assistant. You have access to travel-related "
            "tools. If the user's request cannot be answered using the "
            "available tools, call the decline tool to explain why. "
            "Do NOT call a tool unless it is directly relevant."
        ),
    ),
    user_message="What is the square root of 144?",
    max_iterations=5,
    validate=lambda args: bool(args.get("reason", "").strip()),
    tags=["model_quality"],
    ideal_iterations=1,
)
