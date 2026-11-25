"""Model quality scenarios — tool selection, argument fidelity, reasoning, routing, data gaps."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .._forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario, _check

# ── Pydantic param models ────────────────────────────────────────
# Shared single-field models (reused across scenarios)


class QueryParams(BaseModel):
    query: str = Field(description="The search query")


class PathParams(BaseModel):
    path: str = Field(description="Path to the file")


class CmdParams(BaseModel):
    cmd: str = Field(description="The command to run")


class ToParams(BaseModel):
    to: str = Field(description="Recipient email address")


class NameParams(BaseModel):
    name: str = Field(description="The user's name")


class UserIdParams(BaseModel):
    user_id: str = Field(description="The user's ID")


class AnswerParams(BaseModel):
    answer: str = Field(description="The final answer")


# argument_fidelity


class EntityParams(BaseModel):
    entity: str = Field(description="Entity name")


class EntityIdParams(BaseModel):
    entity_id: str = Field(description="The entity's ID")


class SummaryParams(BaseModel):
    summary: str = Field(description="The summary to present")


# sequential_reasoning


class PatientIdParams(BaseModel):
    patient_id: str = Field(description="The patient's ID")


class PatientNameParams(BaseModel):
    name: str = Field(description="Patient's full name")


class MedicationParams(BaseModel):
    medication: str = Field(description="Medication name")


class RecommendParams(BaseModel):
    patient_id: str = Field(description="The patient's ID")
    findings: str = Field(description="The findings to base the recommendation on")


# conditional_routing


class ServiceParams(BaseModel):
    service: str = Field(description="Service name")


class DiagnoseParams(BaseModel):
    diagnosis: str = Field(description="The root cause diagnosis")
    action: str = Field(description="The recommended action")


# data_gap_recovery


class EmployeeNameParams(BaseModel):
    name: str = Field(description="Employee's full name")


class EmployeeIdParams(BaseModel):
    employee_id: str = Field(description="Employee's ID")


class SubmitReportParams(BaseModel):
    employee_name: str = Field(description="Employee's full name")
    report: str = Field(description="The complete profile report")


# ── Scenario 5: tool_selection ───────────────────────────────────

_tool_selection_tools: dict[str, ToolDef] = {
    "search_web": ToolDef(
        spec=ToolSpec(name="search_web", description="Search the web for information.",
                      parameters=QueryParams),
        callable=lambda **kwargs: "No results found.",
    ),
    "read_file": ToolDef(
        spec=ToolSpec(name="read_file", description="Read a file from disk.",
                      parameters=PathParams),
        callable=lambda **kwargs: "Error: file not found.",
    ),
    "list_directory": ToolDef(
        spec=ToolSpec(name="list_directory", description="List contents of a directory.",
                      parameters=PathParams),
        callable=lambda **kwargs: "Empty directory.",
    ),
    "run_command": ToolDef(
        spec=ToolSpec(name="run_command", description="Run a shell command.",
                      parameters=CmdParams),
        callable=lambda **kwargs: "Permission denied.",
    ),
    "send_email": ToolDef(
        spec=ToolSpec(name="send_email", description="Send an email.",
                      parameters=ToParams),
        callable=lambda **kwargs: "Email sent.",
    ),
    "lookup_user": ToolDef(
        spec=ToolSpec(name="lookup_user", description="Look up a user by name.",
                      parameters=NameParams),
        callable=lambda **kwargs: "User: Alice, Role: Engineer, Team: Platform, ID: U-1001",
    ),
    "get_permissions": ToolDef(
        spec=ToolSpec(name="get_permissions", description="Get permissions for a user by their user ID.",
                      parameters=UserIdParams),
        callable=lambda **kwargs: "Permissions: read, write, admin on repo forge-dev",
    ),
    "respond": ToolDef(
        spec=ToolSpec(name="respond", description="Provide the final answer to the user.",
                      parameters=AnswerParams),
        callable=lambda **kwargs: kwargs.get("answer", ""),
    ),
}

tool_selection = EvalScenario(
    name="tool_selection",
    description="Crowded tool namespace — model must pick correct tools among distractors.",
    workflow=Workflow(
        name="tool_selection",
        description="Look up a user and check permissions",
        tools=_tool_selection_tools,
        required_steps=["lookup_user", "get_permissions"],
        terminal_tool="respond",
        system_prompt_template=(
            "You are an admin assistant. Use the available tools to answer "
            "the user's question. Look up the user first, then check their "
            "permissions, then respond."
        ),
    ),
    user_message="What permissions does Alice have?",
    validate=lambda args: _check(args.get("answer", ""), ["read", "write", "admin"]),
    tags=["model_quality"],
)


# ── Scenario 6: argument_fidelity ────────────────────────────────


def _fetch_details(**kwargs: Any) -> str:
    entity_id = kwargs.get("entity_id", "")
    if entity_id == "ENT-4728":
        return "Details: Widget Pro, created 2024-01-15, 1500 units sold"
    return f"Warning: entity_id '{entity_id}' not recognized, returning default."


_argument_fidelity_tools: dict[str, ToolDef] = {
    "lookup_entity": ToolDef(
        spec=ToolSpec(
            name="lookup_entity",
            description="Look up an entity by name.",
            parameters=EntityParams,
        ),
        callable=lambda **kwargs: "Entity ID: ENT-4728, Status: active, Owner: alice@example.com",
    ),
    "fetch_details": ToolDef(
        spec=ToolSpec(
            name="fetch_details",
            description="Fetch details for an entity by its ID.",
            parameters=EntityIdParams,
        ),
        callable=_fetch_details,
    ),
    "present": ToolDef(
        spec=ToolSpec(
            name="present",
            description="Present the final summary to the user.",
            parameters=SummaryParams,
        ),
        callable=lambda **kwargs: kwargs.get("summary", ""),
    ),
}

argument_fidelity = EvalScenario(
    name="argument_fidelity",
    description="Args from tool results — model must extract entity_id and pass it forward.",
    workflow=Workflow(
        name="argument_fidelity",
        description="Lookup entity, fetch details by ID, present summary",
        tools=_argument_fidelity_tools,
        required_steps=["lookup_entity", "fetch_details"],
        terminal_tool="present",
        system_prompt_template=(
            "You are a helpful assistant. Look up the entity, then fetch "
            "its details using the entity ID from the lookup result, then "
            "present a summary."
        ),
    ),
    user_message="Look up the entity 'Widget Pro' and get its details.",
    validate=lambda args: _check(args.get("summary", ""), ["widget pro", "1500"]),
    tags=["model_quality"],
)


# ── Scenario 7: sequential_reasoning ─────────────────────────────

_sequential_reasoning_tools: dict[str, ToolDef] = {
    "identify_patient": ToolDef(
        spec=ToolSpec(
            name="identify_patient",
            description="Identify a patient by name.",
            parameters=PatientNameParams,
        ),
        callable=lambda **kwargs: "Patient ID: PT-7829, DOB: 1985-03-14, Blood type: O+",
    ),
    "get_records": ToolDef(
        spec=ToolSpec(
            name="get_records",
            description="Get medical records for a patient.",
            parameters=PatientIdParams,
        ),
        callable=lambda **kwargs: (
            "Records: Last visit 2024-11-02, Diagnosis: hypertension, "
            "Medication: lisinopril 10mg"
        ),
    ),
    "check_interactions": ToolDef(
        spec=ToolSpec(
            name="check_interactions",
            description="Check drug interactions for a medication.",
            parameters=MedicationParams,
        ),
        callable=lambda **kwargs: (
            "Interactions: lisinopril + ibuprofen = risk of kidney damage. "
            "lisinopril + potassium supplements = hyperkalemia risk."
        ),
    ),
    "recommend": ToolDef(
        spec=ToolSpec(
            name="recommend",
            description="Provide a recommendation based on findings.",
            parameters=RecommendParams,
        ),
        callable=lambda **kwargs: kwargs.get("findings", ""),
    ),
}


def _validate_sequential_reasoning(args: dict[str, Any]) -> bool:
    text = f"{args.get('findings', '')}".lower()
    has_drug = "lisinopril" in text
    has_kidney = "kidney" in text or "ibuprofen" in text
    has_potassium = "hyperkalemia" in text or "potassium" in text
    return has_drug and has_kidney and has_potassium


sequential_reasoning = EvalScenario(
    name="sequential_reasoning",
    description="4-step chain with data dependency — each tool result informs the next call.",
    workflow=Workflow(
        name="sequential_reasoning",
        description="Identify patient, get records, check interactions, recommend",
        tools=_sequential_reasoning_tools,
        required_steps=["identify_patient", "get_records", "check_interactions"],
        terminal_tool="recommend",
        system_prompt_template=(
            "You are a medical assistant. Identify the patient, retrieve "
            "their records, check drug interactions for their current "
            "medication, then provide a recommendation."
        ),
    ),
    user_message="Check drug interactions for patient John Doe's current medication.",
    validate=_validate_sequential_reasoning,
    tags=["model_quality"],
)


# ── Scenario 8: conditional_routing ─────────────────────────────

_conditional_routing_tools: dict[str, ToolDef] = {
    "get_alert": ToolDef(
        spec=ToolSpec(
            name="get_alert",
            description="Get details for the current P1 alert.",
            parameters=ServiceParams,
        ),
        callable=lambda **kwargs: (
            "Alert: P1-8842 — payments-service\n"
            "Triggered: 2025-01-15 14:23:07 UTC\n"
            "Type: Error Rate Threshold\n"
            "Current error rate: 12.4% (threshold: 2%)\n"
            "Affected endpoint: /api/v2/charge\n"
            "Duration: 18 minutes\n"
            "Last deploy: 2025-01-15 14:04:51 UTC\n"
            "Previous alert on this service: 2024-11-03 (resolved — DB connection pool)"
        ),
    ),
    "check_metrics": ToolDef(
        spec=ToolSpec(
            name="check_metrics",
            description="Get time-series system metrics for a service.",
            parameters=ServiceParams,
        ),
        callable=lambda **kwargs: (
            "Metrics for payments-service (last 60 min):\n"
            "  14:00 — error_rate: 0.3%, latency_p99: 120ms, cpu: 45%, mem: 62%\n"
            "  14:05 — error_rate: 0.4%, latency_p99: 118ms, cpu: 44%, mem: 61%\n"
            "  14:10 — error_rate: 8.1%, latency_p99: 940ms, cpu: 47%, mem: 63%\n"
            "  14:15 — error_rate: 11.2%, latency_p99: 1850ms, cpu: 51%, mem: 64%\n"
            "  14:20 — error_rate: 12.4%, latency_p99: 2100ms, cpu: 52%, mem: 65%\n"
            "  14:25 — error_rate: 12.1%, latency_p99: 2040ms, cpu: 50%, mem: 64%\n"
            "\n"
            "Note: Error spike begins between 14:05 and 14:10. No significant\n"
            "CPU or memory change. Latency correlates with error rate."
        ),
    ),
    "check_logs": ToolDef(
        spec=ToolSpec(
            name="check_logs",
            description="Get recent log entries for a service.",
            parameters=ServiceParams,
        ),
        callable=lambda **kwargs: (
            "Recent logs for payments-service (last 30 min):\n"
            "  14:08:12 WARN  [HttpClient] Retry attempt 1 for upstream call\n"
            "  14:08:15 WARN  [HttpClient] Retry attempt 2 for upstream call\n"
            "  14:09:01 ERROR [PaymentProcessor] Transaction failed: unexpected response format\n"
            "  14:09:03 ERROR [PaymentProcessor] Transaction failed: unexpected response format\n"
            "  14:11:44 WARN  [ConnectionPool] Pool utilization at 78%\n"
            "  14:14:22 ERROR [PaymentProcessor] Transaction failed: unexpected response format\n"
            "  14:18:33 WARN  [HttpClient] Retry attempt 1 for upstream call\n"
            "  (247 similar entries omitted)"
        ),
    ),
    "check_deployment": ToolDef(
        spec=ToolSpec(
            name="check_deployment",
            description="Get details of the last deployment to a service.",
            parameters=ServiceParams,
        ),
        callable=lambda **kwargs: (
            "Last deployment to payments-service:\n"
            "  Deploy ID: deploy-a7f3e2\n"
            "  Timestamp: 2025-01-15 14:04:51 UTC\n"
            "  Author: jenkins-ci (triggered by merge PR #1147)\n"
            "  Changes: Updated payment gateway SDK from v3.8.1 to v4.0.0\n"
            '  Changelog note: "v4.0.0 — Breaking change: response schema updated,\n'
            "    'transaction_id' field moved from root to 'data.transaction_id'\"\n"
            "  Rollback available: Yes (deploy-b82c1a, v3.8.1)"
        ),
    ),
    "diagnose": ToolDef(
        spec=ToolSpec(
            name="diagnose",
            description="Submit a root cause diagnosis and recommended action.",
            parameters=DiagnoseParams,
        ),
        callable=lambda **kwargs: f"Diagnosis: {kwargs.get('diagnosis', '')} | Action: {kwargs.get('action', '')}",
    ),
}


def _validate_conditional_routing(args: dict[str, Any]) -> bool:
    text = f"{args.get('diagnosis', '')} {args.get('action', '')}".lower()
    has_cause = any(t in text for t in ["v4.0.0", "sdk", "gateway"])
    has_action = any(t in text for t in ["rollback", "revert", "roll back"])
    has_mechanism = any(t in text for t in ["response", "schema", "transaction_id"])
    return has_cause and has_action and has_mechanism


conditional_routing = EvalScenario(
    name="conditional_routing",
    description="Incident triage — model must correlate deploy timing with error spike.",
    workflow=Workflow(
        name="conditional_routing",
        description="Diagnose a P1 production incident by correlating alert, metrics, and deployment data",
        tools=_conditional_routing_tools,
        required_steps=["get_alert", "check_metrics"],
        terminal_tool="diagnose",
        system_prompt_template=(
            "You are an infrastructure incident responder. Use the available "
            "tools to investigate the P1 alert, determine the root cause, "
            "and recommend an action. Call diagnose when you have enough "
            "evidence to identify the root cause."
        ),
    ),
    user_message=(
        "We got a P1 alert on the payments service. "
        "Diagnose the root cause and recommend an action."
    ),
    validate=_validate_conditional_routing,
    tags=["model_quality", "reasoning"],
    ideal_iterations=4,
)


# ── Scenario 9: data_gap_recovery ───────────────────────────────

_data_gap_recovery_tools: dict[str, ToolDef] = {
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
            "Clearance Level: [RESTRICTED — request via security_audit]\n"
            "Emergency Contact: [MIGRATED — see historical employee records]"
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
    "personnel_file": ToolDef(
        spec=ToolSpec(
            name="personnel_file",
            description="Access an employee's personnel file.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Personnel File — E-1847 (Sarah Chen)\n"
            "Contains: compensation history, performance reviews, PTO balance.\n"
            "Note: Emergency contact information is not stored in personnel files.\n"
            "It was moved to HR onboarding records during the 2020 system migration."
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
            "Equipment Issued: MacBook Pro 16\", 2x monitors"
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


def _validate_data_gap_recovery(args: dict[str, Any]) -> bool:
    text = f"{args.get('employee_name', '')} {args.get('report', '')}".lower()
    has_employee = "engineering" in text
    has_clearance = "l3" in text or "confidential" in text
    has_contact = "michael" in text and ("867-5309" in text or "spouse" in text)
    return has_employee and has_clearance and has_contact


data_gap_recovery = EvalScenario(
    name="data_gap_recovery",
    description="Dead-end recovery — model must follow hints through misleading tool names.",
    workflow=Workflow(
        name="data_gap_recovery",
        description="Build a complete employee profile by resolving missing data across multiple systems",
        tools=_data_gap_recovery_tools,
        required_steps=["get_employee"],
        terminal_tool="submit_report",
        system_prompt_template=(
            "You are an HR systems assistant. Use the available tools to "
            "gather the requested employee information and submit a complete "
            "profile report."
        ),
    ),
    user_message=(
        "Pull together a complete profile for Sarah Chen — we need her "
        "clearance level and emergency contact for the onboarding audit."
    ),
    validate=_validate_data_gap_recovery,
    tags=["model_quality", "reasoning"],
    ideal_iterations=5,
)

