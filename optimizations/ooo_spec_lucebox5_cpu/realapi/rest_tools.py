"""Real, keyless public REST APIs as read-only tools (BFCL-style multi-step tasks).

Shared by the client agent (rest_bench.py) and the dflash server-side speculative
executor (rest_tool_executor.py) so a speculative hit returns exactly what the
client would have fetched itself. No artificial latency anywhere: every call is a
live HTTPS request to a public API.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from typing import Any

PROTOCOL = "dflash.tool-speculation.v1"
USER_AGENT = "lucebox-toolspec-bench/0.1 (research benchmark; contact davide@cifarelli.tech)"
HTTP_TIMEOUT = 12

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "geocode_city",
        "description": "Look up a city's coordinates with OpenStreetMap Nominatim. Returns latitude/longitude (4 decimals) and display name.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name, optionally with country, e.g. 'Tokyo, Japan'."}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather at coordinates from Open-Meteo: temperature (C), wind speed (km/h), relative humidity, local time.",
        "parameters": {"type": "object", "properties": {
            "latitude": {"type": "number", "description": "Latitude in decimal degrees."},
            "longitude": {"type": "number", "description": "Longitude in decimal degrees."}},
            "required": ["latitude", "longitude"]}}},
    {"type": "function", "function": {
        "name": "country_info",
        "description": "Country facts from the World Bank API: name, capital city, region, income level, latest total population, capital coordinates.",
        "parameters": {"type": "object", "properties": {
            "country_code": {"type": "string", "description": "ISO 3166-1 alpha-2 country code, e.g. 'PE' for Peru."}},
            "required": ["country_code"]}}},
    {"type": "function", "function": {
        "name": "wikipedia_summary",
        "description": "Short summary of an English Wikipedia article (REST v1 page summary).",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Article title, e.g. 'Mount Everest'."}},
            "required": ["title"]}}},
    {"type": "function", "function": {
        "name": "exchange_rate",
        "description": "Latest ECB reference exchange rate between two currencies (Frankfurter API).",
        "parameters": {"type": "object", "properties": {
            "base": {"type": "string", "description": "ISO 4217 code of the base currency, e.g. 'EUR'."},
            "quote": {"type": "string", "description": "ISO 4217 code of the quote currency, e.g. 'JPY'."}},
            "required": ["base", "quote"]}}},
]
TOOL_NAMES = tuple(t["function"]["name"] for t in TOOLS)
READ_ONLY_TOOLS = TOOL_NAMES  # all tools are read-only / idempotent public GETs


def canonical_call(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, separators=(",", ":"))


def call_sha256(name: str, arguments: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_call(name, arguments).encode()).hexdigest()


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "geocode_city":
        city = str(args.get("city", "")).strip()
        data = _get_json("https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": city, "format": "json", "limit": 1}))
        if not data:
            return {"error": f"no result for {city!r}"}
        hit = data[0]
        return {"city": city, "display_name": hit.get("display_name"), "latitude": round(float(hit["lat"]), 4), "longitude": round(float(hit["lon"]), 4)}
    if name == "get_weather":
        lat = float(args.get("latitude")); lon = float(args.get("longitude"))
        data = _get_json("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "current": "temperature_2m,wind_speed_10m,relative_humidity_2m", "timezone": "auto"}))
        cur = data.get("current", {})
        return {"latitude": lat, "longitude": lon, "timezone": data.get("timezone"), "time": cur.get("time"),
                "temperature_c": cur.get("temperature_2m"), "wind_speed_kmh": cur.get("wind_speed_10m"), "relative_humidity_pct": cur.get("relative_humidity_2m")}
    if name == "country_info":
        code = str(args.get("country_code", "")).strip().upper()
        meta = _get_json("https://api.worldbank.org/v2/country/" + urllib.parse.quote(code) + "?format=json")
        if not isinstance(meta, list) or len(meta) < 2 or not meta[1]:
            return {"error": f"country not found: {code!r}"}
        c = meta[1][0]
        pop = _get_json("https://api.worldbank.org/v2/country/" + urllib.parse.quote(code) + "/indicator/SP.POP.TOTL?format=json&mrv=1")
        pop_val = pop_year = None
        if isinstance(pop, list) and len(pop) > 1 and pop[1]:
            pop_val = pop[1][0].get("value"); pop_year = pop[1][0].get("date")
        return {"country_code": code, "name": c.get("name"), "capital": c.get("capitalCity"), "region": (c.get("region") or {}).get("value", "").strip(),
                "income_level": (c.get("incomeLevel") or {}).get("value"), "population": pop_val, "population_year": pop_year,
                "capital_latitude": c.get("latitude"), "capital_longitude": c.get("longitude")}
    if name == "wikipedia_summary":
        title = str(args.get("title", "")).strip().replace(" ", "_")
        data = _get_json("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title))
        return {"title": data.get("title"), "description": data.get("description"), "extract": (data.get("extract") or "")[:700]}
    if name == "exchange_rate":
        base = str(args.get("base", "")).upper().strip(); quote = str(args.get("quote", "")).upper().strip()
        data = _get_json("https://api.frankfurter.app/latest?" + urllib.parse.urlencode({"from": base, "to": quote}))
        rate = (data.get("rates") or {}).get(quote)
        return {"base": base, "quote": quote, "rate": rate, "date": data.get("date")}
    raise ValueError(f"unknown tool {name!r}")


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute one tool call against the live API; returns a raw envelope with timing."""
    started = time.perf_counter()
    try:
        value = _call(name, arguments)
        ok = "error" not in value
    except Exception as exc:  # noqa: BLE001
        value = {"error": f"{type(exc).__name__}: {exc}"}
        ok = False
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {"tool_name": name, "call_sha256": call_sha256(name, arguments), "ok": ok, "value": value, "elapsed_ms": elapsed_ms, "side_effects": False}


def format_result(raw: dict[str, Any]) -> str:
    return json.dumps(raw.get("value"), ensure_ascii=False, separators=(",", ":"))

