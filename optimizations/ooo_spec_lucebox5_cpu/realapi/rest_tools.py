"""Real, keyless public REST APIs as read-only tools (BFCL-style multi-step tasks).

Shared by the client agent (rest_bench.py) and the dflash server-side speculative
executor (rest_tool_executor.py) so a speculative hit returns exactly what the
client would have fetched itself. No artificial latency anywhere: every call is a
live HTTPS request to a public API.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any

PROTOCOL = "dflash.tool-speculation.v1"
USER_AGENT = "lucebox-toolspec-bench/0.1 (research benchmark; contact davide@cifarelli.tech)"
HTTP_TIMEOUT = 12
LIVE_RESULT_MAX_AGE_MS = {
    # These values can change while a long model response is being decoded.
    # The engine checks the attached deadline immediately before committing a
    # speculative result; an expired value falls back to a fresh client call.
    "get_weather": 30_000,
    "exchange_rate": 30_000,
}

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
    result = {"tool_name": name, "call_sha256": call_sha256(name, arguments), "ok": ok, "value": value, "elapsed_ms": elapsed_ms, "side_effects": False}
    if ok and name in LIVE_RESULT_MAX_AGE_MS:
        result["_speculation_fresh_until_unix_ms"] = (
            int(time.time() * 1000) + LIVE_RESULT_MAX_AGE_MS[name]
        )
    return result


def format_result(raw: dict[str, Any]) -> str:
    # Matches nlohmann::json::dump(): compact UTF-8 with object keys ordered.
    # Both benchmark arms therefore feed byte-identical tool messages back to
    # the model, and an early hit also matches prefetch-prefill's cache key.
    return json.dumps(
        raw.get("value"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")
_WIKI_STOPWORDS = {
    "about", "after", "also", "been", "being", "from", "have", "into",
    "more", "most", "over", "that", "their", "there", "these", "they",
    "this", "through", "under", "were", "which", "with", "would",
}


def _numbers(text: str) -> list[float]:
    values = []
    for match in _NUMBER.finditer(text):
        try:
            values.append(float(match.group(0).replace(",", "")))
        except ValueError:
            pass
    return values


def _number_is_reported(final_numbers: list[float], expected: Any) -> bool:
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return False
    expected = float(expected)
    candidates = [expected]
    if abs(expected) >= 1_000:
        candidates.extend((expected / 1_000, expected / 1_000_000,
                           expected / 1_000_000_000))
    return any(
        abs(observed - candidate) <= max(0.5, abs(candidate) * 0.02)
        for observed in final_numbers
        for candidate in candidates
    )


def _string_is_reported(final: str, expected: Any) -> bool:
    return isinstance(expected, str) and bool(expected.strip()) and (
        expected.strip().casefold() in final.casefold()
    )


def _answer_checks_for_result(
    prompt: str, final: str, raw: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build result-derived checks for fields the task actually asks for."""
    name = raw.get("tool_name")
    value = raw.get("value")
    if not isinstance(value, dict):
        return [{"tool": name, "field": "value", "matched": False}]
    prompt_l = prompt.casefold()
    final_numbers = _numbers(final)
    checks: list[dict[str, Any]] = []

    def number(field: str, expected: Any) -> None:
        checks.append({
            "tool": name,
            "field": field,
            "matched": _number_is_reported(final_numbers, expected),
        })

    def string(field: str, expected: Any) -> None:
        checks.append({
            "tool": name,
            "field": field,
            "matched": _string_is_reported(final, expected),
        })

    if name == "get_weather":
        if "wind" in prompt_l:
            number("wind_speed_kmh", value.get("wind_speed_kmh"))
        if "humidity" in prompt_l:
            number("relative_humidity_pct", value.get("relative_humidity_pct"))
        if "time zone" in prompt_l or "timezone" in prompt_l:
            string("timezone", value.get("timezone"))
        if any(term in prompt_l for term in ("temperature", "weather", "warmer")):
            number("temperature_c", value.get("temperature_c"))
    elif name == "country_info":
        if "capital" in prompt_l:
            string("capital", value.get("capital"))
        if any(term in prompt_l for term in ("population", "people live", "how many people")):
            number("population", value.get("population"))
        if "income" in prompt_l:
            string("income_level", value.get("income_level"))
        if "region" in prompt_l:
            string("region", value.get("region"))
    elif name == "exchange_rate":
        rate = value.get("rate")
        matched = _number_is_reported(final_numbers, rate)
        amount_match = re.search(r"\b(\d+(?:\.\d+)?)\b", prompt)
        if amount_match and isinstance(rate, (int, float)):
            amount = float(amount_match.group(1))
            matched = matched or _number_is_reported(final_numbers, amount * rate)
            if rate:
                matched = matched or _number_is_reported(final_numbers, amount / rate)
        checks.append({"tool": name, "field": "rate_or_conversion", "matched": matched})
    elif name == "wikipedia_summary":
        title = value.get("title")
        title_is_new = isinstance(title, str) and title.casefold() not in prompt_l
        title_match = title_is_new and _string_is_reported(final, title)
        source_words = {
            word.casefold()
            for word in _WORD.findall(str(value.get("extract") or ""))
            if word.casefold() not in _WIKI_STOPWORDS
            and word.casefold() not in prompt_l
        }
        final_words = {word.casefold() for word in _WORD.findall(final)}
        checks.append({
            "tool": name,
            "field": "summary_evidence",
            "matched": title_match or len(source_words & final_words) >= 2,
        })
    elif name == "geocode_city" and any(
        term in prompt_l for term in ("latitude", "longitude", "coordinates")
    ):
        number("latitude", value.get("latitude"))
        number("longitude", value.get("longitude"))
    return checks


def validate_answer_from_results(
    prompt: str, final: str | None, results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require successful tools and values derived from every answer-producing result."""
    if not final or not results:
        return {"ok": False, "reason": "missing_final_or_tools", "checks": []}
    if any(not isinstance(raw, dict) or raw.get("ok") is not True for raw in results):
        return {"ok": False, "reason": "tool_failure", "checks": []}
    checks = [
        check
        for raw in results
        for check in _answer_checks_for_result(prompt, final, raw)
    ]
    if not checks:
        return {"ok": False, "reason": "no_result_derived_checks", "checks": []}
    ok = all(check["matched"] for check in checks)
    return {
        "ok": ok,
        "reason": "matched_tool_results" if ok else "tool_result_not_in_answer",
        "checks": checks,
    }
