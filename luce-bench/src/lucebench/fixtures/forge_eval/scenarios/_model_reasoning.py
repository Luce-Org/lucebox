"""Advanced model reasoning scenarios — designed to weed out top-tier
models after the basic model_quality suite saturated post-sampling-fix.

These scenarios layer noise, traps, and multi-hop derivations on top of
the basic capability tests. They share the indirect-validation pattern
(substring-AND on terminal args; downstream tools encode whether the
model derived the right intermediates).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .._forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario
from ._model_quality import (
    EmployeeIdParams,
    EmployeeNameParams,
    SubmitReportParams,
)


# ── Pydantic param models (argument_transformation) ─────────────


class NoParams(BaseModel):
    pass


class QuarterYearParams(BaseModel):
    quarter: str = Field(description="Fiscal quarter, e.g. 'Q4'")
    year: int = Field(description="Fiscal year, e.g. 2024")


class VendorNameParams(BaseModel):
    vendor_name: str = Field(description="Vendor name (case-sensitive in our records)")


class CurrencyConvertParams(BaseModel):
    amount: float = Field(description="Amount to convert")
    from_currency: str = Field(description="Source currency code, e.g. 'EUR'")
    to_currency: str = Field(description="Target currency code, e.g. 'USD'")


class CategorizeExpenseParams(BaseModel):
    amount: float = Field(description="Expense amount")
    category: str = Field(description="Expense category")


class TransactionIdParams(BaseModel):
    transaction_id: str = Field(description="Transaction ID, e.g. 'TX-1001'")


class SubmitAuditReportParams(BaseModel):
    transaction_ids: str = Field(description="Comma-separated transaction IDs to flag")
    total_flagged_usd: str = Field(description="Total flagged amount in USD")
    top_vendor: str = Field(description="Vendor of the single largest flagged transaction")


# ── Scenario: data_gap_recovery_extended ────────────────────────
#
# Extends data_gap_recovery: 5 facts to assemble (vs 3), 2-3 hop chains
# with NO direct tool-name hints (model must discover tools by name and
# infer routing from domain language), 2 misleading-by-name traps
# (compensation_history, access_control), and 1 status-marker lure
# (legacy_records — plausible-but-stale data with an embedded redirect
# note). Plus a soft trap: security_audit returns provisioned access
# groups (payments-prod, internal-apis) that look like a complete answer
# but are missing the v3/v2 suffixes — the model must realize these are
# stale and find project_assignments to get the current versions.
# Same indirect-validation pattern as DGR — the unique markers in the
# report (B7, payments-prod-v3, internal-apis-v2) only appear if the
# model reached the right downstream tools.

_data_gap_recovery_extended_tools: dict[str, ToolDef] = {
    "get_employee": ToolDef(
        spec=ToolSpec(
            name="get_employee",
            description="Look up an employee record by name.",
            parameters=EmployeeNameParams,
        ),
        callable=lambda **kwargs: (
            "Employee Record — Sarah Chen (ID: E-1847)\n"
            "Department: Engineering\n"
            "Title: Senior Backend Engineer\n"
            "Start Date: 2019-03-15\n"
            "Office: Building 3, Floor 2\n"
            "Manager: David Park\n"
            "Clearance Level: [RESTRICTED]\n"
            "Emergency Contact: [unavailable in this record]\n"
            "Salary Band: [PROTECTED]\n"
            "Access Groups: [managed externally]"
        ),
    ),
    "security_audit": ToolDef(
        spec=ToolSpec(
            name="security_audit",
            description="Query security audit records for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Security Audit Record — E-1847 (Sarah Chen)\n"
            "Clearance Level: L3 — Confidential\n"
            "Granted: 2021-06-10\n"
            "Sponsor: David Park (Director, Engineering)\n"
            "Last Review: 2024-12-01 (passed, no findings)\n"
            "Expires: 2025-12-01\n"
            "Access Groups: payments-prod, internal-apis, staging-*"
        ),
    ),
    "onboarding_records": ToolDef(
        spec=ToolSpec(
            name="onboarding_records",
            description="Access HR onboarding records for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "HR Onboarding Record — E-1847 (Sarah Chen)\n"
            "Original Onboarding Date: 2019-03-15\n"
            "Emergency Contact: Michael Chen (spouse) — (555) 867-5309\n"
            "Dietary Restrictions: None\n"
            "T-Shirt Size: M\n"
            'Equipment Issued: MacBook Pro 16", 2x monitors'
        ),
    ),
    "personnel_file": ToolDef(
        spec=ToolSpec(
            name="personnel_file",
            description="Access an employee's personnel file.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Personnel File — E-1847 (Sarah Chen)\n"
            "Contains: performance reviews, PTO balance.\n"
            "Note: Compensation data is in our current pay system (migrated 2024).\n"
            "Note: Emergency contact records are with HR onboarding (2020 migration)."
        ),
    ),
    "compensation_v2": ToolDef(
        spec=ToolSpec(
            name="compensation_v2",
            description="Current compensation system — salary band lookup.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Compensation v2 — E-1847 (Sarah Chen)\n"
            "Salary Band: B7\n"
            "Effective: 2024-04-01\n"
            "Next Review: 2025-04-01"
        ),
    ),
    "project_assignments": ToolDef(
        spec=ToolSpec(
            name="project_assignments",
            description="Current project assignments and active access groups for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Project Assignments — E-1847 (Sarah Chen)\n"
            "Active Access Groups: payments-prod-v3, internal-apis-v2, staging-test\n"
            "Primary Project: Payments Platform Modernization\n"
            "Secondary: Internal Tooling Sprint"
        ),
    ),
    "compensation_history": ToolDef(
        spec=ToolSpec(
            name="compensation_history",
            description="Look up compensation history for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "compensation_history — DEPRECATED endpoint.\n"
            "This service was retired in 2024. Use compensation_v2 for "
            "current salary band data."
        ),
    ),
    "access_control": ToolDef(
        spec=ToolSpec(
            name="access_control",
            description="Look up access control entries for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "access_control — DEPRECATED endpoint.\n"
            "Replaced by project_assignments for current active access "
            "group data. Static provisioning records remain in security_audit."
        ),
    ),
    "legacy_records": ToolDef(
        spec=ToolSpec(
            name="legacy_records",
            description="Search legacy records archive for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Legacy Archive — E-1847 (Sarah Chen)\n"
            "Source: legacy_v1 (READ-ONLY — not synced with current systems)\n"
            "Salary Band: B5\n"
            "Access Groups: legacy-readonly\n"
            "Note: legacy_v1 stopped syncing after the 2024 systems migration. "
            "For current salary band, query compensation_v2. For current access "
            "groups, query project_assignments. Data above is provided for "
            "historical audit only."
        ),
    ),
    "compliance_check": ToolDef(
        spec=ToolSpec(
            name="compliance_check",
            description="Check compliance status for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Compliance Status — E-1847: All mandatory trainings complete.\n"
            "Last security awareness training: 2024-11-15.\n"
            "No outstanding compliance items."
        ),
    ),
    "hr_directory": ToolDef(
        spec=ToolSpec(
            name="hr_directory",
            description="Look up an employee in the HR directory.",
            parameters=EmployeeNameParams,
        ),
        callable=lambda **kwargs: (
            "HR Directory Entry — Sarah Chen\n"
            "Department: Engineering | Reports to: David Park\n"
            "Office: Building 3, Floor 2 | Extension: x4481\n"
            "Status: Active | Full-time"
        ),
    ),
    "submit_report": ToolDef(
        spec=ToolSpec(
            name="submit_report",
            description="Submit a completed employee profile report.",
            parameters=SubmitReportParams,
        ),
        callable=lambda **kwargs: f"Report submitted for {kwargs.get('employee_name', '')}.",
    ),
}


def _validate_data_gap_recovery_extended(args: dict[str, Any]) -> bool:
    text = f"{args.get('employee_name', '')} {args.get('report', '')}".lower()
    has_dept      = "engineering" in text
    has_clearance = "l3" in text or "confidential" in text
    has_contact   = "michael" in text and ("867-5309" in text or "spouse" in text)
    has_salary    = "b7" in text  # only in compensation_v2; "b5" is the legacy_records lure
    has_groups    = "payments-prod-v3" in text or "internal-apis-v2" in text  # only in project_assignments
    return has_dept and has_clearance and has_contact and has_salary and has_groups


data_gap_recovery_extended = EvalScenario(
    name="data_gap_recovery_extended",
    description="Extended hint-chain — 5 facts, 2-3 hop chains, no direct hints, name-trap + status-marker + soft-trap lures.",
    workflow=Workflow(
        name="data_gap_recovery_extended",
        description=(
            "Build a complete employee profile with multi-hop data resolution "
            "across legacy and current systems."
        ),
        tools=_data_gap_recovery_extended_tools,
        required_steps=["get_employee"],
        terminal_tool="submit_report",
        system_prompt_template=(
            "You are an HR systems assistant. Use the available tools to "
            "gather the requested employee information and submit a complete "
            "profile report. Read tool results carefully — some endpoints "
            "redirect to newer systems."
        ),
    ),
    user_message=(
        "Pull together a complete profile for Sarah Chen — we need her "
        "clearance level, current emergency contact, latest salary band, "
        "and active access groups for the onboarding audit and access review."
    ),
    validate=_validate_data_gap_recovery_extended,
    tags=["advanced_reasoning", "reasoning", "model_quality"],
    ideal_iterations=8,
    max_iterations=20,
)


# ── Scenario: argument_transformation ───────────────────────────
#
# Tests reasoning over data, not symbolic plumbing. Tools return
# heterogeneous structured data (transactions in mixed currencies,
# approved-vendor list, vendor aliases); the model must filter,
# convert, disambiguate, and aggregate before calling the terminal
# tool with the derived results.
#
# Top-tier separators (designed to weed out 27B+ models, like DGE):
#  1. Currency conversion: EUR transactions need conversion to USD
#     before applying the $5K threshold. Skipping -> wrong inclusion
#     set + wrong total.
#  2. Vendor case-mismatch: 'ACME Corp' is a registered alias of the
#     approved 'Acme Corp'. Literal string compare over-flags it;
#     get_vendor_details resolves the alias.
#  3. Threshold boundary: '$5,000 or more' -> >=, not >. The $5,000
#     transaction must be included.
#
# Validation pattern (same as DGE / DGR): substring-AND on terminal
# args. Correct filter -> correct ID set + total + top vendor ->
# substring matches succeed. Any reasoning slip -> mismatch -> fail.

# Fixed FX rate for determinism (no live currency lookup).
_FX_EUR_TO_USD = 1.10

_APPROVED_VENDORS = [
    "Acme Corp",
    "Globex Industries",
    "Initech Systems",
    "Umbrella Logistics",
    "Wayne Enterprises",
    "Stark Industries",
]

# Q4 2024 transactions. The 'flag' column is the ground-truth label
# for unit testing; it's not exposed to the model.
_Q4_2024_TRANSACTIONS = [
    # id        date          vendor                amount  currency  flag
    ("TX-1001", "2024-10-05", "Cyberdyne LLC",       7500,  "USD",    True),   # unapproved, >=$5K
    ("TX-1002", "2024-10-12", "Acme Corp",          12000,  "USD",    False),  # approved
    ("TX-1003", "2024-10-22", "Initech Systems",     8200,  "USD",    False),  # approved
    ("TX-1004", "2024-11-03", "Vandelay Imports",    3500,  "USD",    False),  # under $5K
    ("TX-1005", "2024-11-08", "Soylent Corp",        5000,  "USD",    True),   # boundary: $5K exactly
    ("TX-1006", "2024-11-14", "Pied Piper",          4800,  "EUR",    True),   # =$5,280 USD, unapproved
    ("TX-1007", "2024-11-22", "Umbrella Logistics",  9400,  "USD",    False),  # approved
    ("TX-1008", "2024-12-02", "Wonka Industries",   11200,  "USD",    True),   # unapproved, >=$5K, top
    ("TX-1009", "2024-12-08", "ACME Corp",           6500,  "USD",    False),  # case-mismatch alias
    ("TX-1010", "2024-12-15", "Globex Industries",   5500,  "USD",    False),  # approved
    ("TX-1011", "2024-12-19", "Pied Piper",          2400,  "EUR",    False),  # =$2,640 USD, under $5K
    ("TX-1012", "2024-12-22", "Stark Industries",   14800,  "USD",    False),  # approved
    ("TX-1013", "2024-12-28", "Wayne Enterprises",   7300,  "USD",    False),  # approved
]


def _format_transactions(rows: list[tuple[str, str, str, int, str, bool]]) -> str:
    lines = ["Q4 2024 Expense Transactions:"]
    for tid, date, vendor, amount, cur, _flag in rows:
        lines.append(
            f"  {tid} | {date} | vendor: {vendor:25s} | "
            f"amount: {amount:>8,.2f} {cur}"
        )
    return "\n".join(lines)


def _list_transactions(**kwargs: Any) -> str:
    quarter = str(kwargs.get("quarter", "")).strip().upper()
    year = int(kwargs.get("year", 0))
    if quarter == "Q4" and year == 2024:
        return _format_transactions(_Q4_2024_TRANSACTIONS)
    return f"No transactions found for {quarter} {year}."


def _get_approved_vendors(**kwargs: Any) -> str:
    lines = ["Approved Vendors (canonical names — case sensitive):"]
    for v in _APPROVED_VENDORS:
        lines.append(f"  - {v}")
    return "\n".join(lines)


def _get_vendor_details(**kwargs: Any) -> str:
    name = str(kwargs.get("vendor_name", "")).strip()
    # The case-mismatch trap: ACME Corp is the registered alias of Acme Corp.
    if name == "ACME Corp":
        return (
            "Vendor Details — ACME Corp\n"
            "Status: registered trade-name alias of Acme Corp "
            "(unified entity 2023).\n"
            "All purchasing under this name applies to the Acme Corp "
            "master account and is governed by the same approval terms."
        )
    if name == "Acme Corp":
        return (
            "Vendor Details — Acme Corp\n"
            "Status: master account (legal entity since 1998).\n"
            "Trade-name aliases on file: ACME Corp."
        )
    # Other approved vendors return generic stubs.
    if name in _APPROVED_VENDORS:
        return f"Vendor Details — {name}\nStatus: standard supplier (active)."
    # Unknown / unapproved vendors return a no-record stub. No useful info,
    # but doesn't break the model — it can still flag based on the approved
    # list directly.
    return f"Vendor Details — {name}\nStatus: not found in vendor master."


def _currency_convert(**kwargs: Any) -> str:
    amount = float(kwargs.get("amount", 0))
    src = str(kwargs.get("from_currency", "")).strip().upper()
    dst = str(kwargs.get("to_currency", "")).strip().upper()
    if src == "EUR" and dst == "USD":
        converted = amount * _FX_EUR_TO_USD
        return (
            f"Conversion: {amount:,.2f} EUR = {converted:,.2f} USD "
            f"(rate: 1 EUR = {_FX_EUR_TO_USD} USD)"
        )
    if src == "USD" and dst == "EUR":
        converted = amount / _FX_EUR_TO_USD
        return (
            f"Conversion: {amount:,.2f} USD = {converted:,.2f} EUR "
            f"(rate: 1 EUR = {_FX_EUR_TO_USD} USD)"
        )
    if src == dst:
        return f"Conversion: {amount:,.2f} {src} = {amount:,.2f} {dst} (same currency)."
    return f"Unsupported conversion pair: {src} -> {dst}."


def _categorize_expense(**kwargs: Any) -> str:
    # Distractor: returns a category bucket, no useful audit info.
    amount = float(kwargs.get("amount", 0))
    category = str(kwargs.get("category", "")).strip()
    return (
        f"Categorization: {amount:,.2f} -> {category} "
        f"(GL bucket: GL-{abs(hash(category)) % 9000 + 1000})."
    )


def _lookup_transaction(**kwargs: Any) -> str:
    # Redundant helper: returns a single transaction's record. Distractor —
    # list_transactions already returns everything.
    tid = str(kwargs.get("transaction_id", "")).strip()
    for row in _Q4_2024_TRANSACTIONS:
        if row[0] == tid:
            _id, date, vendor, amount, cur, _flag = row
            return (
                f"Transaction Record — {tid}\n"
                f"Date: {date}\nVendor: {vendor}\n"
                f"Amount: {amount:,.2f} {cur}"
            )
    return f"No transaction found for '{tid}'."


_argument_transformation_tools: dict[str, ToolDef] = {
    "list_transactions": ToolDef(
        spec=ToolSpec(
            name="list_transactions",
            description="List all expense transactions for a given fiscal quarter and year.",
            parameters=QuarterYearParams,
        ),
        callable=_list_transactions,
    ),
    "get_approved_vendors": ToolDef(
        spec=ToolSpec(
            name="get_approved_vendors",
            description="Return the canonical list of approved vendor names (case sensitive).",
            parameters=NoParams,
        ),
        callable=_get_approved_vendors,
    ),
    "get_vendor_details": ToolDef(
        spec=ToolSpec(
            name="get_vendor_details",
            description="Look up vendor details (status, legal entity, trade-name aliases).",
            parameters=VendorNameParams,
        ),
        callable=_get_vendor_details,
    ),
    "currency_convert": ToolDef(
        spec=ToolSpec(
            name="currency_convert",
            description="Convert an amount between currencies (USD/EUR).",
            parameters=CurrencyConvertParams,
        ),
        callable=_currency_convert,
    ),
    "categorize_expense": ToolDef(
        spec=ToolSpec(
            name="categorize_expense",
            description="Assign an expense to a GL category bucket.",
            parameters=CategorizeExpenseParams,
        ),
        callable=_categorize_expense,
    ),
    "lookup_transaction": ToolDef(
        spec=ToolSpec(
            name="lookup_transaction",
            description="Look up a single transaction's full record by ID.",
            parameters=TransactionIdParams,
        ),
        callable=_lookup_transaction,
    ),
    "submit_audit_report": ToolDef(
        spec=ToolSpec(
            name="submit_audit_report",
            description=(
                "Submit the completed audit report. Provide comma-separated "
                "transaction IDs of flagged items, the total flagged amount in "
                "USD, and the vendor of the single largest flagged transaction."
            ),
            parameters=SubmitAuditReportParams,
        ),
        callable=lambda **kwargs: (
            f"Audit report submitted. "
            f"Flagged: {kwargs.get('transaction_ids', '')}; "
            f"total: {kwargs.get('total_flagged_usd', '')}; "
            f"top: {kwargs.get('top_vendor', '')}."
        ),
    ),
}


# Expected ground truth (used only for validation):
#  Flagged: TX-1001 ($7,500), TX-1005 ($5,000), TX-1006 ($5,280 from €4,800),
#           TX-1008 ($11,200) -> total $28,980; top vendor Wonka Industries.
_ARGTRANS_REQUIRED_IDS = ("TX-1001", "TX-1005", "TX-1006", "TX-1008")
_ARGTRANS_TOTAL_TOKENS = ("28,980", "28980")  # accept either format
_ARGTRANS_TOP_VENDOR = "wonka"


def _validate_argument_transformation(args: dict[str, Any]) -> bool:
    ids_text = str(args.get("transaction_ids", ""))
    total_text = str(args.get("total_flagged_usd", "")).replace("$", "")
    vendor_text = str(args.get("top_vendor", "")).lower()

    has_all_ids = all(tid in ids_text for tid in _ARGTRANS_REQUIRED_IDS)
    has_total = any(tok in total_text for tok in _ARGTRANS_TOTAL_TOKENS)
    has_vendor = _ARGTRANS_TOP_VENDOR in vendor_text
    return has_all_ids and has_total and has_vendor


argument_transformation = EvalScenario(
    name="argument_transformation",
    description=(
        "Argument transformation — filter + currency-convert + vendor "
        "disambiguation + aggregate over structured tool returns."
    ),
    workflow=Workflow(
        name="argument_transformation",
        description=(
            "Run a Q4 expense audit by deriving the flagged-transaction "
            "set, total in USD, and top vendor from heterogeneous tool "
            "data, then submitting the audit report."
        ),
        tools=_argument_transformation_tools,
        required_steps=["list_transactions", "get_approved_vendors"],
        terminal_tool="submit_audit_report",
        system_prompt_template=(
            "You are an expense audit assistant. Use the available tools "
            "to identify flagged transactions and submit a complete audit "
            "report. Read tool results carefully — amounts may be in "
            "different currencies and vendor records may have aliases."
        ),
    ),
    user_message=(
        "Run our Q4 2024 expense audit. Flag any transaction of $5,000 "
        "or more from vendors NOT on our approved list. Submit the audit "
        "report with: comma-separated transaction IDs, total flagged "
        "amount in USD, and the vendor of the single largest flagged "
        "transaction."
    ),
    validate=_validate_argument_transformation,
    tags=["advanced_reasoning", "reasoning", "model_quality"],
    ideal_iterations=5,
    max_iterations=15,
)


# ── Scenario: inconsistent_api_recovery ─────────────────────────
#
# Tests cascading error recovery across heterogeneously-designed legacy
# APIs. Each tool violates a different convention dimension (ID format,
# date format, units, casing, separator, encoding) — so the recovery
# lesson from one tool is wrong for the next. Designed for the
# guardrails-vs-bare ablation: bare runs hit the first error and
# cascade; reforged runs use retry nudges to recover at each step.
#
# Convention violations across the chain (no transferable lesson):
#  Step 1 list_accounts:       page/page_size pagination; int IDs
#  Step 2 get_balance:         ACC-prefixed string IDs (not int from #1);
#                              returns cents + Unix timestamps
#  Step 3 get_transactions:    ISO date strings (not the Unix ts from #2);
#                              returns TXN/NNNNN string IDs
#  Step 4 categorize_spend:    int txn_id (extract from "TXN/00042");
#                              UPPERCASE 4-letter category code
#  Step 5 check_compliance:    LOWERCASE 2-letter region (case flips
#                              from #4); ISO quarter notation "2024-Q4"
#  Step 6 aggregate_subtotal:  pipe-separated decimal dollars (not list,
#                              not cents — units flip back from #2)
#  Step 7 submit_audit:        JSON-encoded string (not natural dict)


class PageParams(BaseModel):
    page: int = Field(description="Page number")
    page_size: int = Field(description="Records per page")


class AccountIdParams(BaseModel):
    account_id: str = Field(description="Account identifier")


class TransactionRangeParams(BaseModel):
    account_id: str = Field(description="Account identifier")
    since: str = Field(description="Range start")
    until: str = Field(description="Range end")


class CategorizeSpendParams(BaseModel):
    txn_id: int = Field(description="Transaction identifier")
    category: str = Field(description="Spend category")


class ComplianceCheckParams(BaseModel):
    region: str = Field(description="Region")
    period: str = Field(description="Reporting period")


class AggregateSubtotalParams(BaseModel):
    amounts: str = Field(description="Amounts to sum")


class SubmitAuditParams(BaseModel):
    report: str = Field(description="Audit report")


_LEGACY_ACCOUNTS = [
    (12345, "Acme Corp"),
    (67890, "Globex Industries"),
    (24680, "Initech Systems"),
]

_LEGACY_BALANCES = {
    "ACC-12345": {"amount_cents": 750000, "last_txn_unix": 1696000000, "status": "ACTIVE"},
    "ACC-67890": {"amount_cents": 320000, "last_txn_unix": 1697500000, "status": "ACTIVE"},
    "ACC-24680": {"amount_cents": 180000, "last_txn_unix": 1698200000, "status": "FROZEN"},
}

_LEGACY_TRANSACTIONS_ACME = [
    ("TXN/00042", "2024-10-05",  500000, "services"),
    ("TXN/00043", "2024-11-12", 1250000, "hardware"),
    ("TXN/00044", "2024-12-08",  800000, "services"),
]

_VALID_CATEGORY_CODES = {"SVCS", "HRDW", "TRVL", "MISC"}
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_QUARTER_RE = re.compile(r"^\d{4}-Q[1-4]$")


def _legacy_list_accounts(**kwargs: Any) -> str:
    page = kwargs.get("page")
    page_size = kwargs.get("page_size")
    if page is None or page_size is None:
        return (
            "ERROR: legacy_list_accounts requires 'page' and 'page_size' "
            "(not offset/limit)."
        )
    try:
        p = int(page)
        ps = int(page_size)
    except (TypeError, ValueError):
        return "ERROR: page and page_size must be integers."
    if p < 1 or ps < 1:
        return "ERROR: page and page_size must be >= 1."
    n = len(_LEGACY_ACCOUNTS)
    lines = [f"Page {p} of 1 ({n} of {n} accounts):"]
    for acc_id, name in _LEGACY_ACCOUNTS:
        lines.append(f"  - id: {acc_id} | name: {name}")
    return "\n".join(lines)


def _legacy_get_balance(**kwargs: Any) -> str:
    aid = str(kwargs.get("account_id", "")).strip()
    if not aid.startswith("ACC-"):
        return (
            f"ERROR: account_id '{aid}' must include the 'ACC-' prefix "
            "(e.g. 'ACC-12345')."
        )
    if aid not in _LEGACY_BALANCES:
        return f"ERROR: account '{aid}' not found in balance system."
    b = _LEGACY_BALANCES[aid]
    return (
        f"Balance for {aid}: amount={b['amount_cents']} (cents); "
        f"last_txn={b['last_txn_unix']} (unix); status={b['status']}"
    )


def _legacy_get_transactions(**kwargs: Any) -> str:
    aid = str(kwargs.get("account_id", "")).strip()
    since = str(kwargs.get("since", "")).strip()
    until = str(kwargs.get("until", "")).strip()
    if not aid.startswith("ACC-"):
        return (
            f"ERROR: account_id '{aid}' must include the 'ACC-' prefix "
            "(e.g. 'ACC-12345')."
        )
    if not _ISO_DATE_RE.match(since) or not _ISO_DATE_RE.match(until):
        return (
            f"ERROR: since/until must be ISO date format YYYY-MM-DD "
            f"(got since='{since}', until='{until}'). Unix timestamps are "
            "not accepted here."
        )
    if aid != "ACC-12345":
        return f"No transactions on file for {aid} between {since} and {until}."
    lines = [f"Transactions for {aid} ({since} to {until}):"]
    for tid, date, amount_cents, category in _LEGACY_TRANSACTIONS_ACME:
        lines.append(
            f"  {tid} | {date} | amount: {amount_cents} (cents) | "
            f"category: {category}"
        )
    return "\n".join(lines)


def _legacy_categorize_spend(**kwargs: Any) -> str:
    raw_txn = kwargs.get("txn_id")
    cat = str(kwargs.get("category", "")).strip()
    if isinstance(raw_txn, str) and not raw_txn.isdigit():
        return (
            f"ERROR: txn_id '{raw_txn}' must be the numeric component as int "
            "(for 'TXN/00042' pass 42, not the full string)."
        )
    try:
        txn_id = int(raw_txn)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return (
            "ERROR: txn_id must be an integer (for 'TXN/00042' pass 42)."
        )
    if cat not in _VALID_CATEGORY_CODES:
        return (
            f"ERROR: category '{cat}' must be uppercase 4-letter code: "
            "SVCS|HRDW|TRVL|MISC."
        )
    for tid, _date, amount_cents, _cat in _LEGACY_TRANSACTIONS_ACME:
        if tid == f"TXN/{txn_id:05d}":
            return (
                f"Categorized TXN/{txn_id:05d} ({cat}): "
                f"amount={amount_cents/100:.2f} USD; bucket=GL-2400."
            )
    return f"Categorized txn_id={txn_id} ({cat}): no amount on record."


def _legacy_check_compliance(**kwargs: Any) -> str:
    region = str(kwargs.get("region", "")).strip()
    period = str(kwargs.get("period", "")).strip()
    if not (len(region) == 2 and region.islower() and region.isalpha()):
        return (
            f"ERROR: region '{region}' must be lowercase 2-letter ISO code "
            "(e.g. 'us', 'gb')."
        )
    if not _ISO_QUARTER_RE.match(period):
        return (
            f"ERROR: period '{period}' must be ISO quarter notation "
            "YYYY-QN (e.g. '2024-Q4'). Dates and Unix timestamps are not "
            "accepted here."
        )
    return (
        f"Compliance status for region={region}, period={period}: PASS "
        "(3 checks: AML/KYC/SOX); flagged_count=0."
    )


def _legacy_aggregate_subtotal(**kwargs: Any) -> str:
    s = str(kwargs.get("amounts", "")).strip()
    if not s:
        return "ERROR: amounts is empty."
    if "|" not in s:
        return (
            f"ERROR: amounts '{s}' must be pipe-separated decimal dollars "
            "(e.g. '1000.00|2500.00'). Lists, comma/space-separated values, "
            "and cent values are not accepted."
        )
    parts = [p.strip() for p in s.split("|") if p.strip()]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return (
            f"ERROR: amounts '{s}' contains non-numeric values; expected "
            "decimal dollars."
        )
    total = sum(nums)
    return f"Subtotal: {total:.2f} USD ({len(nums)} amounts processed)."


def _legacy_submit_audit(**kwargs: Any) -> str:
    s = str(kwargs.get("report", "")).strip()
    return f"Audit submitted. Report: {s}"


_inconsistent_api_recovery_tools: dict[str, ToolDef] = {
    "legacy_list_accounts": ToolDef(
        spec=ToolSpec(
            name="legacy_list_accounts",
            description="List available accounts.",
            parameters=PageParams,
        ),
        callable=_legacy_list_accounts,
    ),
    "legacy_get_balance": ToolDef(
        spec=ToolSpec(
            name="legacy_get_balance",
            description="Get the current balance for an account.",
            parameters=AccountIdParams,
        ),
        callable=_legacy_get_balance,
    ),
    "legacy_get_transactions": ToolDef(
        spec=ToolSpec(
            name="legacy_get_transactions",
            description="List transactions for an account over a date range.",
            parameters=TransactionRangeParams,
        ),
        callable=_legacy_get_transactions,
    ),
    "legacy_categorize_spend": ToolDef(
        spec=ToolSpec(
            name="legacy_categorize_spend",
            description="Assign a spend category to a transaction.",
            parameters=CategorizeSpendParams,
        ),
        callable=_legacy_categorize_spend,
    ),
    "legacy_check_compliance": ToolDef(
        spec=ToolSpec(
            name="legacy_check_compliance",
            description="Run a regional compliance check for a reporting period.",
            parameters=ComplianceCheckParams,
        ),
        callable=_legacy_check_compliance,
    ),
    "legacy_aggregate_subtotal": ToolDef(
        spec=ToolSpec(
            name="legacy_aggregate_subtotal",
            description="Compute the subtotal of a set of amounts.",
            parameters=AggregateSubtotalParams,
        ),
        callable=_legacy_aggregate_subtotal,
    ),
    "legacy_submit_audit": ToolDef(
        spec=ToolSpec(
            name="legacy_submit_audit",
            description="Submit the final compliance audit report.",
            parameters=SubmitAuditParams,
        ),
        callable=_legacy_submit_audit,
    ),
}


# Canonical answer for ACC-12345 (Acme Corp) Q4 2024:
#   3 transactions @ $5,000 + $12,500 + $8,000 = $25,500 USD
#   Compliance: PASS, 0 flagged
#   txn_count: 3
_INCAPI_REQUIRED_TOKENS = ("25500", "pass", "txn_count")


def _validate_inconsistent_api_recovery(args: dict[str, Any]) -> bool:
    text = str(args.get("report", "")).lower().replace(",", "")
    return all(tok in text for tok in _INCAPI_REQUIRED_TOKENS)


inconsistent_api_recovery = EvalScenario(
    name="inconsistent_api_recovery",
    description=(
        "Cascading error recovery across legacy APIs with no transferable "
        "convention — each tool violates a different format/encoding/units rule."
    ),
    workflow=Workflow(
        name="inconsistent_api_recovery",
        description=(
            "Run a Q4 2024 compliance audit on a legacy account by chaining "
            "seven inconsistently-designed APIs."
        ),
        tools=_inconsistent_api_recovery_tools,
        required_steps=["legacy_list_accounts"],
        terminal_tool="legacy_submit_audit",
        system_prompt_template=(
            "You are a compliance audit assistant. Use the available "
            "tools to complete the requested audit."
        ),
    ),
    user_message=(
        "Run a Q4 2024 (Oct 1 - Dec 31) compliance audit on Acme Corp "
        "(account ACC-12345). Pull the account balance and Q4 transactions, "
        "categorize at least one transaction, run a US-region compliance "
        "check for the period, calculate the subtotal of all transaction "
        "amounts in USD, and submit the audit report with keys: "
        "flagged_count (int), total_usd (str), compliance_status (str), "
        "txn_count (int)."
    ),
    validate=_validate_inconsistent_api_recovery,
    tags=["advanced_reasoning", "reasoning", "error_recovery"],
    ideal_iterations=8,
    max_iterations=20,
)


# ── Scenario: grounded_synthesis ────────────────────────────────
#
# Tests synthesis under attractor pressure: the model receives pre-
# distilled assessments from sub-agent tools and must produce a
# recommendation that respects hard constraints surfaced in one tool's
# output, even when other tools' outputs all push toward an attractor
# choice that violates the constraint. Mirrors the failure mode
# observed in 40kAIo's Commander pattern (small models recommended
# loading GRAVIS units into Impulsors despite a clear keyword exclusion).
#
# Pool design (4 candidates, 2+2 split):
#   Sarah Chen   — top-decile attractor; BLOCKER: Stripe non-compete
#   James Patel  — top-decile attractor; BLOCKER: visa sponsorship gap
#   Aisha Nakamura — eligible, payments-experienced (CORRECT pick)
#   Marcus Reyes  — eligible, weaker preference (less exp, no payments)
#
# Three failure tiers (graceful degradation, not binary):
#   1. Picks Sarah/James -> ignored hard constraint (worst)
#   2. Picks Marcus      -> filtered constraints, missed preference (mid)
#   3. Picks Aisha       -> synthesized correctly (success)
#
# Validator (lambda): substring check on terminal candidate + rationale.


class CandidateNameParams(BaseModel):
    candidate_name: str = Field(description="Candidate full name")


class SubmitHiringDecisionParams(BaseModel):
    candidate: str = Field(description="Selected candidate's name")
    rationale: str = Field(description="Brief justification for the decision")


_GS_OPEN_ROLE = (
    "Open Role: Senior Backend Engineer, Payments Platform.\n"
    "Requirements:\n"
    "  - 5+ years Python production experience\n"
    "  - Distributed systems background\n"
    "  - Payment processing experience strongly preferred (we are "
    "building Stripe-Connect-style merchant integrations)\n"
    "Context: this hire is for the Q3 product launch — the team needs "
    "the new senior IC in seat and ramped well before the launch sprint "
    "kicks off."
)


_GS_CANDIDATES: dict[str, dict[str, str]] = {
    "Sarah Chen": {
        "brief": "8y Python, ex-Stripe Senior Engineer, payments domain leader",
        "skill_summary": (
            "Sarah Chen — Skill Assessment\n"
            "Years: 8 | Primary stack: Python, Go, PostgreSQL, Redis\n"
            "Domain depth: Payments — shipped Stripe Connect's webhook "
            "reliability layer; designed idempotency middleware for Stripe "
            "Subscriptions\n"
            "Technical interview: Top decile (98th percentile)\n"
            "Portfolio: 4 production payment-routing systems\n"
            "Mentorship: led team of 4 at Stripe; promoted 2 junior "
            "engineers to mid-level"
        ),
        "compatibility": (
            "Compatibility — Sarah Chen\n"
            "Background check: clean. References: glowing.\n"
            "Note from Legal: Sarah's prior employment with Stripe included "
            "a standard payments-vertical non-compete covenant that does "
            "not lapse until late 2026. Starting her on a payments product "
            "before then would require Legal to negotiate a waiver with "
            "Stripe. Based on past similar cases, that's a meaningful "
            "project of its own — drawn-out, with no guarantee of success. "
            "Alternatively, she could start in a non-payments role "
            "(infrastructure or platform) where the covenant does not "
            "apply.\n"
            "Compensation expectation: senior band, market-competitive."
        ),
        "team_dynamics": (
            "Team Dynamics — Sarah Chen + Payments Platform team\n"
            "Existing team has a notable gap in webhook reliability and "
            "idempotency expertise (currently relying on a third-party SaaS "
            "vendor). Sarah's Stripe webhook background closes this gap "
            "precisely. Cultural fit: very strong — peers describe her as "
            "'the engineer you want in the room when payment edge cases "
            "come up'. Estimated ramp: 2 weeks to first production change."
        ),
    },
    "James Patel": {
        "brief": "9y Python, ex-DeepMind infrastructure, distributed-systems expert",
        "skill_summary": (
            "James Patel — Skill Assessment\n"
            "Years: 9 | Primary stack: Python, C++, distributed systems\n"
            "Domain depth: ML training infrastructure — designed DeepMind's "
            "gradient-aggregation tier; contributed to JAX runtime\n"
            "Technical interview: Top decile (96th percentile)\n"
            "Portfolio: 3 large-scale ML training platforms; less direct "
            "payments experience but very strong distributed-systems "
            "intuition\n"
            "Mentorship: led infra team of 6 at DeepMind"
        ),
        "compatibility": (
            "Compatibility — James Patel\n"
            "Background check: clean. References: glowing.\n"
            "Note from People Ops: James is currently on F-1 OPT; for him "
            "to continue working past his current authorization, our "
            "company would need to sponsor him for an H1B visa. We do "
            "sponsor H1B in selected engineering tracks, but the Payments "
            "Platform team has not been allocated a sponsorship slot in "
            "this hiring cycle. Bringing James onto Payments Platform "
            "would require petitioning the immigration committee for an "
            "additional slot — historically a slow process with mixed "
            "success.\n"
            "Compensation expectation: senior band, market-competitive."
        ),
        "team_dynamics": (
            "Team Dynamics — James Patel + Payments Platform team\n"
            "The team's distributed-systems intuition is uneven; James "
            "would lift the bar on consistency reasoning at scale. He is "
            "distributed-systems first and payments second — about 4-6 "
            "weeks of payments-domain learning curve. Cultural fit: very "
            "strong — references describe him as a 'systems thinker who "
            "lifts everyone around him'."
        ),
    },
    "Aisha Nakamura": {
        "brief": "7y Python, ex-Square, payments-adjacent backend",
        "skill_summary": (
            "Aisha Nakamura — Skill Assessment\n"
            "Years: 7 | Primary stack: Python, Java, MySQL, Kafka\n"
            "Domain depth: Payments backend at Square — built reconciliation "
            "pipeline; owned merchant-payout retries\n"
            "Technical interview: Strong (78th percentile)\n"
            "Portfolio: 2 payments-adjacent backend systems\n"
            "Mentorship: tech-lead for team of 3"
        ),
        "compatibility": (
            "Compatibility — Aisha Nakamura\n"
            "Background check: clean. References: solid.\n"
            "Note from People Ops: US citizen. No outstanding non-competes "
            "or restrictive covenants. Standard onboarding paperwork — she "
            "could start essentially immediately upon offer acceptance.\n"
            "Compensation expectation: senior band, market-competitive."
        ),
        "team_dynamics": (
            "Team Dynamics — Aisha Nakamura + Payments Platform team\n"
            "Brings payments-experienced senior immediately. Reconciliation "
            "and retry expertise from Square is directly applicable. "
            "Cultural fit: strong — references describe her as steady and "
            "pragmatic. Estimated ramp: 2-3 weeks."
        ),
    },
    "Marcus Reyes": {
        "brief": "5y Python, ex-Adobe Creative Cloud, generalist backend",
        "skill_summary": (
            "Marcus Reyes — Skill Assessment\n"
            "Years: 5 | Primary stack: Python, JavaScript, MongoDB\n"
            "Domain depth: Generalist backend at Adobe — built licensing "
            "and entitlement APIs for Creative Cloud\n"
            "Technical interview: Solid (72nd percentile)\n"
            "Portfolio: 1 backend system at scale (Creative Cloud "
            "entitlements); no payments background\n"
            "Mentorship: senior IC, no formal management experience"
        ),
        "compatibility": (
            "Compatibility — Marcus Reyes\n"
            "Background check: clean. References: solid.\n"
            "Note from People Ops: US permanent resident. No outstanding "
            "non-competes or restrictive covenants. Adobe IP-assignment "
            "standard release on file. He could start essentially "
            "immediately upon offer acceptance.\n"
            "Compensation expectation: senior band, market-competitive."
        ),
        "team_dynamics": (
            "Team Dynamics — Marcus Reyes + Payments Platform team\n"
            "Adds a generalist senior. Payments domain learning curve is "
            "significant — Adobe licensing and entitlement is structurally "
            "different from payment processing. Cultural fit: strong — "
            "references describe him as adaptable and curious. Estimated "
            "ramp: 8-12 weeks (substantial payments-domain learning)."
        ),
    },
    "Diana Kim": {
        "brief": "10y Python, ex-Stripe Principal Engineer, payments domain expert",
        "skill_summary": (
            "Diana Kim — Skill Assessment\n"
            "Years: 10 | Primary stack: Python, Go, PostgreSQL, gRPC\n"
            "Domain depth: Payments — Principal Engineer at Stripe; led "
            "Stripe Issuing's authorization service; co-designed the "
            "global fraud-scoring pipeline\n"
            "Technical interview: Top decile (99th percentile)\n"
            "Portfolio: 5 production payments systems at scale; multiple "
            "internal-tech-blog publications\n"
            "Mentorship: led Stripe's payments-platform reading group; "
            "mentored 6 senior ICs"
        ),
        "compatibility": (
            "Compatibility — Diana Kim\n"
            "Background check: clean. References: glowing.\n"
            "Note from Legal: Diana's prior employment with Stripe "
            "predates their current non-compete policy; her offer letter "
            "contained no restrictive covenants. US citizen. Standard "
            "onboarding paperwork — she could start essentially "
            "immediately upon offer acceptance.\n"
            "Compensation expectation: principal band; she has indicated "
            "flexibility to accept senior-band comp for the right role."
        ),
        "team_dynamics": (
            "Team Dynamics — Diana Kim + Payments Platform team\n"
            "Diana would be a strong addition technically — Stripe "
            "Issuing experience directly applies to our roadmap. However, "
            "in initial conversations she made clear that what she is "
            "primarily looking for in her next move is a step into a "
            "tech-lead or engineering-management track with direct "
            "reports — she is evaluating multiple offers, several of "
            "which are people-management roles. The Senior Backend "
            "Engineer position is an IC track without direct reports, "
            "and we have no open lead positions on Payments Platform "
            "this cycle. The recruiter's read from her conversations is "
            "that she would take this role only if her preferred lead "
            "offers fall through, and would likely keep an eye out for "
            "lead opportunities elsewhere over time."
        ),
    },
}


def _gs_lookup(name: str) -> str | None:
    """Resolve a candidate name match (case-insensitive, partial OK)."""
    key = name.strip().lower()
    if not key:
        return None
    for canonical in _GS_CANDIDATES:
        clow = canonical.lower()
        if key == clow or key in clow.split() or key in clow:
            return canonical
    return None


def _gs_get_open_role(**kwargs: Any) -> str:
    return _GS_OPEN_ROLE


def _gs_get_candidate_pool(**kwargs: Any) -> str:
    n = len(_GS_CANDIDATES)
    lines = [f"Candidate Pool — Senior Backend Engineer, Payments Platform ({n} candidates):"]
    for name, info in _GS_CANDIDATES.items():
        lines.append(f"  - {name} — {info['brief']}")
    return "\n".join(lines)


def _gs_get_skill_summary(**kwargs: Any) -> str:
    name = str(kwargs.get("candidate_name", "")).strip()
    canonical = _gs_lookup(name)
    if not canonical:
        return f"No candidate found matching '{name}'."
    return _GS_CANDIDATES[canonical]["skill_summary"]


def _gs_get_compatibility_check(**kwargs: Any) -> str:
    name = str(kwargs.get("candidate_name", "")).strip()
    canonical = _gs_lookup(name)
    if not canonical:
        return f"No candidate found matching '{name}'."
    return _GS_CANDIDATES[canonical]["compatibility"]


def _gs_get_team_dynamics(**kwargs: Any) -> str:
    name = str(kwargs.get("candidate_name", "")).strip()
    canonical = _gs_lookup(name)
    if not canonical:
        return f"No candidate found matching '{name}'."
    return _GS_CANDIDATES[canonical]["team_dynamics"]


def _gs_submit_hiring_decision(**kwargs: Any) -> str:
    candidate = str(kwargs.get("candidate", "")).strip()
    rationale = str(kwargs.get("rationale", "")).strip()
    return f"Hiring decision recorded. Candidate: {candidate}. Rationale: {rationale}"


_grounded_synthesis_tools: dict[str, ToolDef] = {
    "get_open_role": ToolDef(
        spec=ToolSpec(
            name="get_open_role",
            description="Get the spec and constraints for the open role.",
            parameters=NoParams,
        ),
        callable=_gs_get_open_role,
    ),
    "get_candidate_pool": ToolDef(
        spec=ToolSpec(
            name="get_candidate_pool",
            description="List the candidates being considered.",
            parameters=NoParams,
        ),
        callable=_gs_get_candidate_pool,
    ),
    "get_skill_summary": ToolDef(
        spec=ToolSpec(
            name="get_skill_summary",
            description="Get a candidate's distilled skill assessment.",
            parameters=CandidateNameParams,
        ),
        callable=_gs_get_skill_summary,
    ),
    "get_compatibility_check": ToolDef(
        spec=ToolSpec(
            name="get_compatibility_check",
            description="Check a candidate's compliance and eligibility for the open role.",
            parameters=CandidateNameParams,
        ),
        callable=_gs_get_compatibility_check,
    ),
    "get_team_dynamics": ToolDef(
        spec=ToolSpec(
            name="get_team_dynamics",
            description="Get the team-fit analysis for a candidate.",
            parameters=CandidateNameParams,
        ),
        callable=_gs_get_team_dynamics,
    ),
    "submit_hiring_decision": ToolDef(
        spec=ToolSpec(
            name="submit_hiring_decision",
            description="Submit the final hiring decision with a brief rationale.",
            parameters=SubmitHiringDecisionParams,
        ),
        callable=_gs_submit_hiring_decision,
    ),
}


_GS_CORRECT_CANDIDATE_TOKENS = ("aisha", "nakamura")
_GS_BLOCKER_TOKENS = (
    # Sarah: non-compete covenant with Stripe (waiver process)
    "non-compete", "noncompete", "non compete", "covenant", "waiver",
    # James: visa sponsorship gap
    "visa", "sponsor", "f-1", "h1b", "h-1b", "opt", "immigration",
    "work auth",
    # Diana: role mismatch (looking for tech-lead/EM, this is IC)
    "tech-lead", "tech lead", "engineering manager", "engineering management",
    "management track", "lead role", "lead position", "lead opportunit",
    "ic track", "ic role", "fallback", "people-management",
)
# Alternative pass path: model articulated rejection by naming the
# unsuccessful candidates. Catches well-reasoned rationales that use
# synonyms not in the blocker token list above.
_GS_REJECTED_NAMES = ("sarah", "chen", "james", "patel", "diana", "kim")


def _validate_grounded_synthesis(args: dict[str, Any]) -> bool:
    candidate = str(args.get("candidate", "")).lower()
    rationale = str(args.get("rationale", "")).lower()
    has_correct_pick = any(tok in candidate for tok in _GS_CORRECT_CANDIDATE_TOKENS)
    has_blocker_ref = any(tok in rationale for tok in _GS_BLOCKER_TOKENS)
    has_rejection_named = any(name in rationale for name in _GS_REJECTED_NAMES)
    return has_correct_pick and (has_blocker_ref or has_rejection_named)


grounded_synthesis = EvalScenario(
    name="grounded_synthesis",
    description=(
        "Synthesis under attractor pressure — model must respect hard "
        "constraints surfaced in one tool's output even when other tools "
        "all reinforce a violating choice."
    ),
    workflow=Workflow(
        name="grounded_synthesis",
        description=(
            "Pick the right candidate for an open role from a pool of four, "
            "honoring compliance constraints surfaced by sub-agent tools."
        ),
        tools=_grounded_synthesis_tools,
        required_steps=["get_open_role", "get_candidate_pool"],
        terminal_tool="submit_hiring_decision",
        system_prompt_template=(
            "You are a hiring manager. Use the available tools to evaluate "
            "the candidate pool for the open role and submit your decision."
        ),
    ),
    user_message=(
        "We have an open senior backend engineering role and four candidates "
        "in the pool. Review the role, evaluate the candidates, and submit "
        "your hiring decision with a brief rationale explaining why you "
        "selected (and did not select) the candidates you considered."
    ),
    validate=_validate_grounded_synthesis,
    tags=["advanced_reasoning", "reasoning", "model_quality"],
    ideal_iterations=10,
    max_iterations=20,
)
