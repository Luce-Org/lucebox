"""Stateful relevance detection — verifies no travel tools were called."""

from __future__ import annotations

from .._forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario, _placeholder_workflow
from ._compaction import (
    CityParams,
    CurrencyParams,
    FlightParams,
    HotelParams,
    ReasonParams,
)


class TravelBookingSystem:
    def __init__(self) -> None:
        self.forecasts_fetched: list[str] = []
        self.flights_booked: list[tuple[str, str]] = []
        self.hotels_checked: list[str] = []
        self.conversions: list[tuple[str, str, str]] = []

    def get_forecast(self, city: str) -> str:
        self.forecasts_fetched.append(city.strip().lower())
        return f"Forecast for {city}: 22°C, partly cloudy."

    def book_flight(self, origin: str, destination: str) -> str:
        self.flights_booked.append(
            (origin.strip().lower(), destination.strip().lower())
        )
        return f"Flight booked: {origin} → {destination}."

    def check_hotel(self, city: str, checkin: str) -> str:
        self.hotels_checked.append(city.strip().lower())
        return f"3 hotels available in {city} for {checkin}."

    def convert_currency(
        self, amount: str, from_currency: str, to_currency: str,
    ) -> str:
        self.conversions.append(
            (amount, from_currency.upper(), to_currency.upper())
        )
        return f"{amount} {from_currency} = 0.00 {to_currency}."

    def decline(self, reason: str) -> str:
        return reason


def _build_relevance_detection_stateful() -> tuple[Workflow, callable]:
    db = TravelBookingSystem()
    tools: dict[str, ToolDef] = {
        "get_forecast": ToolDef(
            spec=ToolSpec(
                name="get_forecast",
                description="Get weather forecast for a city.",
                parameters=CityParams,
            ),
            callable=lambda **kw: db.get_forecast(kw["city"]),
        ),
        "book_flight": ToolDef(
            spec=ToolSpec(
                name="book_flight",
                description="Book a flight between two cities.",
                parameters=FlightParams,
            ),
            callable=lambda **kw: db.book_flight(kw["origin"], kw["destination"]),
        ),
        "check_hotel": ToolDef(
            spec=ToolSpec(
                name="check_hotel",
                description="Check hotel availability in a city.",
                parameters=HotelParams,
            ),
            callable=lambda **kw: db.check_hotel(kw["city"], kw["checkin"]),
        ),
        "convert_currency": ToolDef(
            spec=ToolSpec(
                name="convert_currency",
                description="Convert an amount between currencies.",
                parameters=CurrencyParams,
            ),
            callable=lambda **kw: db.convert_currency(
                kw["amount"], kw["from_currency"], kw["to_currency"],
            ),
        ),
        "decline": ToolDef(
            spec=ToolSpec(
                name="decline",
                description="Call this when none of the available tools are relevant to the user's request.",
                parameters=ReasonParams,
            ),
            callable=lambda **kw: db.decline(kw.get("reason", "")),
        ),
    }
    workflow = Workflow(
        name="relevance_detection_stateful",
        description="Travel tools that are irrelevant to the user's question",
        tools=tools,
        required_steps=[],
        terminal_tool="decline",
        system_prompt_template=(
            "You are a helpful assistant. You have access to travel-related "
            "tools. If the user's request cannot be answered using the "
            "available tools, call the decline tool to explain why. "
            "Do NOT call a tool unless it is directly relevant."
        ),
    )
    validate_state = lambda: (
        db.forecasts_fetched == []
        and db.flights_booked == []
        and db.hotels_checked == []
        and db.conversions == []
    )
    return workflow, validate_state


relevance_detection_stateful = EvalScenario(
    name="relevance_detection_stateful",
    description="Stateful relevance detection — verifies no travel tools were called.",
    workflow=_placeholder_workflow("relevance_detection_stateful", "decline"),
    user_message="What is the square root of 144?",
    max_iterations=5,
    validate=lambda args: bool(args.get("reason", "").strip()),
    build_workflow=_build_relevance_detection_stateful,
    tags=["stateful", "model_quality"],
    ideal_iterations=1,
)
