"""Stateful model quality scenarios — tool selection, argument fidelity,
sequential reasoning, conditional routing, data gap recovery."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .._forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario, _check, _placeholder_workflow


# ── Pydantic parameter models ──────────────────────────────────


# Shared single-field models
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

class EntityParams(BaseModel):
    entity: str = Field(description="Entity name")

class EntityIdParams(BaseModel):
    entity_id: str = Field(description="The entity's ID")

class SummaryParams(BaseModel):
    summary: str = Field(description="The summary to present")

class PatientNameParams(BaseModel):
    name: str = Field(description="Patient's full name")

class PatientIdParams(BaseModel):
    patient_id: str = Field(description="The patient's ID")

class MedicationParams(BaseModel):
    medication: str = Field(description="Medication name")

class PatientRecommendParams(BaseModel):
    patient_id: str = Field(description="The patient's ID")
    findings: str = Field(description="The findings to base the recommendation on")

class ServiceParams(BaseModel):
    service: str = Field(description="Service name")

class DiagnoseParams(BaseModel):
    diagnosis: str = Field(description="The root cause diagnosis")
    action: str = Field(description="The recommended action")

class EmployeeNameParams(BaseModel):
    name: str = Field(description="Employee's full name")

class EmployeeIdParams(BaseModel):
    employee_id: str = Field(description="Employee's ID")

class SubmitReportParams(BaseModel):
    employee_name: str = Field(description="Employee's full name")
    report: str = Field(description="The complete profile report")


# ── Backend 4: UserPermissionsDB ────────────────────────────────


class UserPermissionsDB:
    def __init__(self) -> None:
        self.users = {
            "alice": {"role": "Engineer", "team": "Platform", "user_id": "U-1001"},
            "bob": {"role": "Manager", "team": "Infrastructure", "user_id": "U-1002"},
        }
        self.permissions = {
            "U-1001": ["read", "write", "admin"],
            "U-1002": ["read"],
        }
        self.looked_up: dict[str, str] | None = None
        self.perms_fetched: list[str] | None = None

    def lookup_user(self, name: str) -> str:
        key = name.strip().lower()
        if key in self.users:
            u = self.users[key]
            self.looked_up = u
            return (
                f"User: {name.title()}, Role: {u['role']}, "
                f"Team: {u['team']}, ID: {u['user_id']}"
            )
        return f"No user found for '{name}'."

    def get_permissions(self, user_id: str) -> str:
        uid = user_id.strip()
        if uid in self.permissions:
            perms = self.permissions[uid]
            self.perms_fetched = perms
            return f"Permissions: {', '.join(perms)} on repo forge-dev"
        return f"No permissions found for '{user_id}'."

    def respond(self, answer: str) -> str:
        return answer  # echo-back terminal


def _build_tool_selection_stateful() -> tuple[Workflow, callable]:
    db = UserPermissionsDB()

    # 6 distractors — same as lambda version
    distractors: dict[str, ToolDef] = {
        "search_web": ToolDef(
            spec=ToolSpec(name="search_web", description="Search the web for information.",
                          parameters=QueryParams),
            callable=lambda **kw: "No results found.",
        ),
        "read_file": ToolDef(
            spec=ToolSpec(name="read_file", description="Read a file from disk.",
                          parameters=PathParams),
            callable=lambda **kw: "Error: file not found.",
        ),
        "list_directory": ToolDef(
            spec=ToolSpec(name="list_directory", description="List contents of a directory.",
                          parameters=PathParams),
            callable=lambda **kw: "Empty directory.",
        ),
        "run_command": ToolDef(
            spec=ToolSpec(name="run_command", description="Run a shell command.",
                          parameters=CmdParams),
            callable=lambda **kw: "Permission denied.",
        ),
        "send_email": ToolDef(
            spec=ToolSpec(name="send_email", description="Send an email.",
                          parameters=ToParams),
            callable=lambda **kw: "Email sent.",
        ),
    }

    real_tools: dict[str, ToolDef] = {
        "lookup_user": ToolDef(
            spec=ToolSpec(
                name="lookup_user",
                description="Look up a user by name.",
                parameters=NameParams,
            ),
            callable=lambda **kw: db.lookup_user(kw["name"]),
        ),
        "get_permissions": ToolDef(
            spec=ToolSpec(
                name="get_permissions",
                description="Get permissions for a user by their user ID.",
                parameters=UserIdParams,
            ),
            callable=lambda **kw: db.get_permissions(kw["user_id"]),
        ),
        "respond": ToolDef(
            spec=ToolSpec(
                name="respond",
                description="Provide the final answer to the user.",
                parameters=AnswerParams,
            ),
            callable=lambda **kw: db.respond(kw.get("answer", "")),
        ),
    }

    tools = {**distractors, **real_tools}
    workflow = Workflow(
        name="tool_selection_stateful",
        description="Look up a user and check permissions (stateful, with distractors)",
        tools=tools,
        required_steps=["lookup_user", "get_permissions"],
        terminal_tool="respond",
        system_prompt_template=(
            "You are an admin assistant. Use the available tools to answer "
            "the user's question. Look up the user first, then check their "
            "permissions, then respond."
        ),
    )
    validate_state = lambda: (
        db.looked_up is not None
        and db.perms_fetched == ["read", "write", "admin"]
    )
    return workflow, validate_state


tool_selection_stateful = EvalScenario(
    name="tool_selection_stateful",
    description="Stateful tool selection — ID threading through crowded namespace.",
    workflow=_placeholder_workflow("tool_selection_stateful", "respond", ["lookup_user", "get_permissions"]),
    user_message="What permissions does Alice have?",
    validate=lambda args: _check(args.get("answer", ""), ["read", "write", "admin"]),
    build_workflow=_build_tool_selection_stateful,
    tags=["stateful", "model_quality"],
    ideal_iterations=3,
)


# ── Backend 5: EntityRegistry ───────────────────────────────────


class EntityRegistry:
    def __init__(self) -> None:
        self.entities = {
            "widget pro": {
                "entity_id": "ENT-4728", "status": "active",
                "owner": "alice@example.com",
            },
            "widget basic": {
                "entity_id": "ENT-3301", "status": "retired",
                "owner": "bob@example.com",
            },
        }
        self.details = {
            "ENT-4728": {
                "name": "Widget Pro", "created": "2024-01-15",
                "units_sold": 1500,
            },
            "ENT-3301": {
                "name": "Widget Basic", "created": "2022-06-01",
                "units_sold": 800,
            },
        }
        self.looked_up: dict[str, str] | None = None
        self.fetched: dict[str, Any] | None = None

    def lookup_entity(self, entity: str) -> str:
        key = entity.strip().lower()
        if key in self.entities:
            e = self.entities[key]
            self.looked_up = e
            return (
                f"Entity ID: {e['entity_id']}, "
                f"Status: {e['status']}, Owner: {e['owner']}"
            )
        return f"No entity found for '{entity}'."

    def fetch_details(self, entity_id: str) -> str:
        eid = entity_id.strip()
        if eid in self.details:
            d = self.details[eid]
            self.fetched = d
            return (
                f"Details: {d['name']}, created {d['created']}, "
                f"{d['units_sold']} units sold"
            )
        return f"No details found for entity ID '{entity_id}'."

    def present(self, summary: str) -> str:
        return summary  # echo-back terminal


def _build_argument_fidelity_stateful() -> tuple[Workflow, callable]:
    db = EntityRegistry()
    tools: dict[str, ToolDef] = {
        "lookup_entity": ToolDef(
            spec=ToolSpec(
                name="lookup_entity",
                description="Look up an entity by name.",
                parameters=EntityParams,
            ),
            callable=lambda **kw: db.lookup_entity(kw["entity"]),
        ),
        "fetch_details": ToolDef(
            spec=ToolSpec(
                name="fetch_details",
                description="Fetch details for an entity by its ID.",
                parameters=EntityIdParams,
            ),
            callable=lambda **kw: db.fetch_details(kw["entity_id"]),
        ),
        "present": ToolDef(
            spec=ToolSpec(
                name="present",
                description="Present the final summary to the user.",
                parameters=SummaryParams,
            ),
            callable=lambda **kw: db.present(kw.get("summary", "")),
        ),
    }
    workflow = Workflow(
        name="argument_fidelity_stateful",
        description="Lookup entity, fetch details by ID, present summary",
        tools=tools,
        required_steps=["lookup_entity", "fetch_details"],
        terminal_tool="present",
        system_prompt_template=(
            "You are a helpful assistant. Look up the entity, then fetch "
            "its details using the entity ID from the lookup result, then "
            "present a summary."
        ),
    )
    validate_state = lambda: (
        db.looked_up is not None
        and db.fetched is not None
        and db.fetched["name"] == "Widget Pro"
    )
    return workflow, validate_state


argument_fidelity_stateful = EvalScenario(
    name="argument_fidelity_stateful",
    description="Stateful argument fidelity — alphanumeric ID threading with decoy entity.",
    workflow=_placeholder_workflow("argument_fidelity_stateful", "present", ["lookup_entity", "fetch_details"]),
    user_message="Look up the entity 'Widget Pro' and get its details.",
    validate=lambda args: _check(args.get("summary", ""), ["widget pro", "1500"]),
    build_workflow=_build_argument_fidelity_stateful,
    tags=["stateful", "model_quality"],
    ideal_iterations=3,
)


# ── Backend 6: MedicalRecordsDB ────────────────────────────────


class MedicalRecordsDB:
    def __init__(self) -> None:
        self.patients = {
            "john doe": "PT-7829",
            "jane smith": "PT-4215",
        }
        # Return strings match lambda scenario exactly
        self.patient_info = {
            "PT-7829": "Patient ID: PT-7829, DOB: 1985-03-14, Blood type: O+",
            "PT-4215": "Patient ID: PT-4215, DOB: 1990-07-22, Blood type: A-",
        }
        self.record_info = {
            "PT-7829": (
                "Records: Last visit 2024-11-02, Diagnosis: hypertension, "
                "Medication: lisinopril 10mg"
            ),
            "PT-4215": (
                "Records: Last visit 2024-10-18, Diagnosis: type 2 diabetes, "
                "Medication: metformin 500mg"
            ),
        }
        self.interaction_info = {
            "lisinopril": (
                "Interactions: lisinopril + ibuprofen = risk of kidney damage. "
                "lisinopril + potassium supplements = hyperkalemia risk."
            ),
            "metformin": (
                "Interactions: metformin + alcohol = lactic acidosis risk. "
                "metformin + contrast dye = kidney injury risk."
            ),
        }
        self.identified: str | None = None
        self.records_fetched: str | None = None
        self.interactions_checked: str | None = None

    def identify_patient(self, name: str) -> str:
        key = name.strip().lower()
        if key in self.patients:
            pid = self.patients[key]
            self.identified = pid
            return self.patient_info[pid]
        return f"No patient found for '{name}'."

    def get_records(self, patient_id: str) -> str:
        pid = patient_id.strip()
        if pid in self.record_info:
            self.records_fetched = pid
            return self.record_info[pid]
        return f"No records found for patient ID '{patient_id}'."

    def check_interactions(self, medication: str) -> str:
        # "lisinopril 10mg" → "lisinopril"
        key = medication.strip().lower().split()[0]
        if key in self.interaction_info:
            self.interactions_checked = key
            return self.interaction_info[key]
        return f"No interaction data found for '{medication}'."

    def recommend(self, patient_id: str, findings: str) -> str:
        return findings  # echo-back terminal


def _validate_sequential_reasoning_stateful(args: dict[str, Any]) -> bool:
    text = f"{args.get('findings', '')}".lower()
    has_drug = "lisinopril" in text
    has_kidney = "kidney" in text or "ibuprofen" in text
    has_potassium = "hyperkalemia" in text or "potassium" in text
    return has_drug and has_kidney and has_potassium


def _build_sequential_reasoning_stateful() -> tuple[Workflow, callable]:
    db = MedicalRecordsDB()
    tools: dict[str, ToolDef] = {
        "identify_patient": ToolDef(
            spec=ToolSpec(
                name="identify_patient",
                description="Identify a patient by name.",
                parameters=PatientNameParams,
            ),
            callable=lambda **kw: db.identify_patient(kw["name"]),
        ),
        "get_records": ToolDef(
            spec=ToolSpec(
                name="get_records",
                description="Get medical records for a patient.",
                parameters=PatientIdParams,
            ),
            callable=lambda **kw: db.get_records(kw["patient_id"]),
        ),
        "check_interactions": ToolDef(
            spec=ToolSpec(
                name="check_interactions",
                description="Check drug interactions for a medication.",
                parameters=MedicationParams,
            ),
            callable=lambda **kw: db.check_interactions(kw["medication"]),
        ),
        "recommend": ToolDef(
            spec=ToolSpec(
                name="recommend",
                description="Provide a recommendation based on findings.",
                parameters=PatientRecommendParams,
            ),
            callable=lambda **kw: db.recommend(kw.get("patient_id", ""), kw.get("findings", "")),
        ),
    }
    workflow = Workflow(
        name="sequential_reasoning_stateful",
        description="Identify patient, get records, check interactions, recommend",
        tools=tools,
        required_steps=["identify_patient", "get_records", "check_interactions"],
        terminal_tool="recommend",
        system_prompt_template=(
            "You are a medical assistant. Identify the patient, retrieve "
            "their records, check drug interactions for their current "
            "medication, then provide a recommendation."
        ),
    )
    validate_state = lambda: (
        db.identified == "PT-7829"
        and db.records_fetched == "PT-7829"
        and db.interactions_checked == "lisinopril"
    )
    return workflow, validate_state


sequential_reasoning_stateful = EvalScenario(
    name="sequential_reasoning_stateful",
    description="Stateful 4-step chain — three-link data dependency with decoy patient.",
    workflow=_placeholder_workflow("sequential_reasoning_stateful", "recommend", ["identify_patient", "get_records", "check_interactions"]),
    user_message="Check drug interactions for patient John Doe's current medication.",
    validate=_validate_sequential_reasoning_stateful,
    build_workflow=_build_sequential_reasoning_stateful,
    tags=["stateful", "model_quality"],
    ideal_iterations=4,
)


# ── Backend 7: IncidentTriage ──────────────────────────────────


class IncidentTriage:
    def __init__(self) -> None:
        self.alerts = {
            "payments-service": {
                "alert_id": "P1-8842", "type": "Error Rate Threshold",
                "error_rate": "12.4%", "threshold": "2%",
                "endpoint": "/api/v2/charge", "duration_min": 18,
                "triggered": "2025-01-15 14:23:07 UTC",
                "last_deploy": "2025-01-15 14:04:51 UTC",
                "previous_alert": "2024-11-03 (resolved — DB connection pool)",
            },
            "auth-service": {
                "alert_id": "P2-3317", "type": "Latency Threshold",
                "error_rate": "0.8%", "threshold": "500ms",
                "endpoint": "/api/v1/login", "duration_min": 5,
                "triggered": "2025-01-14 09:15:00 UTC",
                "last_deploy": "2025-01-10 11:00:00 UTC",
                "previous_alert": "2024-12-20 (resolved — rate limiter tuning)",
            },
        }
        self.metrics = {
            "payments-service": (
                "Metrics for payments-service (last 60 min):\n"
                "  14:00 — error_rate: 0.3%, latency_p99: 120ms, "
                "cpu: 45%, mem: 62%\n"
                "  14:05 — error_rate: 0.4%, latency_p99: 118ms, "
                "cpu: 44%, mem: 61%\n"
                "  14:10 — error_rate: 8.1%, latency_p99: 940ms, "
                "cpu: 47%, mem: 63%\n"
                "  14:15 — error_rate: 11.2%, latency_p99: 1850ms, "
                "cpu: 51%, mem: 64%\n"
                "  14:20 — error_rate: 12.4%, latency_p99: 2100ms, "
                "cpu: 52%, mem: 65%\n"
                "  14:25 — error_rate: 12.1%, latency_p99: 2040ms, "
                "cpu: 50%, mem: 64%\n"
                "\n"
                "Note: Error spike begins between 14:05 and 14:10. No significant\n"
                "CPU or memory change. Latency correlates with error rate."
            ),
            "auth-service": (
                "Metrics for auth-service (last 60 min):\n"
                "  09:10 — error_rate: 0.1%, latency_p99: 80ms, "
                "cpu: 30%, mem: 45%\n"
                "  09:15 — error_rate: 0.8%, latency_p99: 620ms, "
                "cpu: 32%, mem: 46%\n"
                "  09:20 — error_rate: 0.2%, latency_p99: 95ms, "
                "cpu: 30%, mem: 45%\n"
                "\n"
                "Note: Brief latency spike at 09:15, self-resolved "
                "within 5 minutes."
            ),
        }
        self.logs = {
            "payments-service": (
                "Recent logs for payments-service (last 30 min):\n"
                "  14:08:12 WARN  [HttpClient] Retry attempt 1 for "
                "upstream call\n"
                "  14:08:15 WARN  [HttpClient] Retry attempt 2 for "
                "upstream call\n"
                "  14:09:01 ERROR [PaymentProcessor] Transaction failed: "
                "unexpected response format\n"
                "  14:09:03 ERROR [PaymentProcessor] Transaction failed: "
                "unexpected response format\n"
                "  14:11:44 WARN  [ConnectionPool] Pool utilization "
                "at 78%\n"
                "  14:14:22 ERROR [PaymentProcessor] Transaction failed: "
                "unexpected response format\n"
                "  14:18:33 WARN  [HttpClient] Retry attempt 1 for "
                "upstream call\n"
                "  (247 similar entries omitted)"
            ),
            "auth-service": (
                "Recent logs for auth-service (last 30 min):\n"
                "  09:14:55 WARN  [RateLimiter] Spike in login attempts "
                "from subnet 10.0.3.0/24\n"
                "  09:15:01 WARN  [RateLimiter] Throttling enabled\n"
                "  09:15:42 INFO  [RateLimiter] Throttling cleared, "
                "traffic normal\n"
                "  (3 entries total)"
            ),
        }
        self.deploys = {
            "payments-service": (
                "Last deployment to payments-service:\n"
                "  Deploy ID: deploy-a7f3e2\n"
                "  Timestamp: 2025-01-15 14:04:51 UTC\n"
                "  Author: jenkins-ci (triggered by merge PR #1147)\n"
                "  Changes: Updated payment gateway SDK from v3.8.1 "
                "to v4.0.0\n"
                '  Changelog note: "v4.0.0 — Breaking change: response '
                "schema updated, 'transaction_id' field moved from root "
                "to 'data.transaction_id'\"\n"
                "  Rollback available: Yes (deploy-b82c1a, v3.8.1)"
            ),
            "auth-service": (
                "Last deployment to auth-service:\n"
                "  Deploy ID: deploy-c4d9f1\n"
                "  Timestamp: 2025-01-10 11:00:00 UTC\n"
                "  Author: jenkins-ci (triggered by merge PR #1098)\n"
                "  Changes: Updated logging library to v2.1.0\n"
                "  Rollback available: Yes (deploy-d1e2a3)"
            ),
        }
        self.alert_checked: str | None = None
        self.metrics_checked: str | None = None
        self.logs_checked: str | None = None
        self.deploy_checked: str | None = None

    def _resolve_service(self, service: str) -> str | None:
        """Normalize service name with fuzzy matching.

        Accepts 'payments', 'payments-service', 'the payments service', etc.
        """
        key = service.strip().lower()
        if key in self.alerts:
            return key
        # Try appending '-service'
        candidate = key + "-service"
        if candidate in self.alerts:
            return candidate
        # Try stripping common prefixes/suffixes: "the payments service" → "payments"
        for word in ("the ", "service"):
            key = key.replace(word, "").strip()
        if key in self.alerts:
            return key
        candidate = key + "-service"
        if candidate in self.alerts:
            return candidate
        return None

    def get_alert(self, service: str) -> str:
        key = self._resolve_service(service)
        if key is not None:
            a = self.alerts[key]
            self.alert_checked = key
            return (
                f"Alert: {a['alert_id']} — {key}\n"
                f"Triggered: {a['triggered']}\n"
                f"Type: {a['type']}\n"
                f"Current error rate: {a['error_rate']} "
                f"(threshold: {a['threshold']})\n"
                f"Affected endpoint: {a['endpoint']}\n"
                f"Duration: {a['duration_min']} minutes\n"
                f"Last deploy: {a['last_deploy']}\n"
                f"Previous alert on this service: {a['previous_alert']}"
            )
        return f"No alert found for service '{service}'."

    def check_metrics(self, service: str) -> str:
        key = self._resolve_service(service)
        if key is not None:
            self.metrics_checked = key
            return self.metrics[key]
        return f"No metrics found for service '{service}'."

    def check_logs(self, service: str) -> str:
        key = self._resolve_service(service)
        if key is not None:
            self.logs_checked = key
            return self.logs[key]
        return f"No logs found for service '{service}'."

    def check_deployment(self, service: str) -> str:
        key = self._resolve_service(service)
        if key is not None:
            self.deploy_checked = key
            return self.deploys[key]
        return f"No deployment info for service '{service}'."

    def diagnose(self, diagnosis: str, action: str) -> str:
        return f"Diagnosis: {diagnosis} | Action: {action}"


def _validate_conditional_routing_stateful(args: dict[str, Any]) -> bool:
    text = f"{args.get('diagnosis', '')} {args.get('action', '')}".lower()
    has_cause = any(t in text for t in ["v4.0.0", "sdk", "gateway"])
    has_action = any(t in text for t in ["rollback", "revert", "roll back"])
    has_mechanism = any(t in text for t in ["response", "schema", "transaction_id"])
    return has_cause and has_action and has_mechanism


def _build_conditional_routing_stateful() -> tuple[Workflow, callable]:
    db = IncidentTriage()
    tools: dict[str, ToolDef] = {
        "get_alert": ToolDef(
            spec=ToolSpec(
                name="get_alert",
                description="Get details for the current P1 alert.",
                parameters=ServiceParams,
            ),
            callable=lambda **kw: db.get_alert(kw["service"]),
        ),
        "check_metrics": ToolDef(
            spec=ToolSpec(
                name="check_metrics",
                description="Get time-series system metrics for a service.",
                parameters=ServiceParams,
            ),
            callable=lambda **kw: db.check_metrics(kw["service"]),
        ),
        "check_logs": ToolDef(
            spec=ToolSpec(
                name="check_logs",
                description="Get recent log entries for a service.",
                parameters=ServiceParams,
            ),
            callable=lambda **kw: db.check_logs(kw["service"]),
        ),
        "check_deployment": ToolDef(
            spec=ToolSpec(
                name="check_deployment",
                description="Get details of the last deployment to a service.",
                parameters=ServiceParams,
            ),
            callable=lambda **kw: db.check_deployment(kw["service"]),
        ),
        "diagnose": ToolDef(
            spec=ToolSpec(
                name="diagnose",
                description="Submit a root cause diagnosis and recommended action.",
                parameters=DiagnoseParams,
            ),
            callable=lambda **kw: db.diagnose(kw.get("diagnosis", ""), kw.get("action", "")),
        ),
    }
    workflow = Workflow(
        name="conditional_routing_stateful",
        description="Diagnose a P1 incident by correlating alert, metrics, and deployment data",
        tools=tools,
        required_steps=["get_alert", "check_metrics"],
        terminal_tool="diagnose",
        system_prompt_template=(
            "You are an infrastructure incident responder. Use the available "
            "tools to investigate the P1 alert, determine the root cause, "
            "and recommend an action. Call diagnose when you have enough "
            "evidence to identify the root cause."
        ),
    )
    validate_state = lambda: (
        db.alert_checked == "payments-service"
        and db.metrics_checked == "payments-service"
        and db.deploy_checked == "payments-service"
    )
    return workflow, validate_state


conditional_routing_stateful = EvalScenario(
    name="conditional_routing_stateful",
    description="Stateful incident triage — service param selects between two incident patterns.",
    workflow=_placeholder_workflow("conditional_routing_stateful", "diagnose", ["get_alert", "check_metrics"]),
    user_message=(
        "We got a P1 alert on the payments service. "
        "Diagnose the root cause and recommend an action."
    ),
    validate=_validate_conditional_routing_stateful,
    build_workflow=_build_conditional_routing_stateful,
    tags=["stateful", "model_quality", "reasoning"],
    ideal_iterations=4,
)


# ── Backend 8: HRRecordsSystem ──────────────────────────────────


class HRRecordsSystem:
    def __init__(self) -> None:
        self.employees = {
            "sarah chen": {
                "employee_id": "E-1847", "department": "Engineering",
                "title": "Senior Backend Engineer",
                "start_date": "2019-03-15",
                "office": "Building 3, Floor 2",
                "extension": "x4481",
                "manager": "David Park",
            },
            "james liu": {
                "employee_id": "E-2234", "department": "Marketing",
                "title": "Content Strategist",
                "start_date": "2021-08-01",
                "office": "Building 1, Floor 4",
                "extension": "x5592",
                "manager": "Maria Santos",
            },
        }
        self.security = {
            "E-1847": {
                "name": "Sarah Chen",
                "clearance": "L3 — Confidential",
                "granted": "2021-06-10",
                "sponsor": "David Park (Director, Engineering)",
                "last_review": "2024-12-01 (passed, no findings)",
                "expires": "2025-12-01",
                "access_groups": "payments-prod, internal-apis, staging-*",
            },
            "E-2234": {
                "name": "James Liu",
                "clearance": "L1 — Public",
                "granted": "2021-08-01",
                "sponsor": "Maria Santos (VP, Marketing)",
                "last_review": "2024-11-15 (passed, no findings)",
                "expires": "2026-08-01",
                "access_groups": "marketing-tools, cms-prod",
            },
        }
        self.onboarding = {
            "E-1847": {
                "name": "Sarah Chen",
                "onboarding_date": "2019-03-15",
                "emergency_contact": "Michael Chen (spouse) — (555) 867-5309",
                "dietary": "None",
                "tshirt": "M",
                "equipment": 'MacBook Pro 16", 2x monitors',
            },
            "E-2234": {
                "name": "James Liu",
                "onboarding_date": "2021-08-01",
                "emergency_contact": "Linda Liu (mother) — (555) 234-5678",
                "dietary": "Vegetarian",
                "tshirt": "L",
                "equipment": 'MacBook Pro 14", 1x monitor',
            },
        }
        self.employee_looked_up: dict[str, str] | None = None
        self.security_fetched: str | None = None
        self.onboarding_fetched: str | None = None

    def get_employee(self, name: str) -> str:
        key = name.strip().lower()
        if key in self.employees:
            e = self.employees[key]
            self.employee_looked_up = e
            return (
                f"Employee Record — {name.title()} "
                f"(ID: {e['employee_id']})\n"
                f"Department: {e['department']}\n"
                f"Title: {e['title']}\n"
                f"Start Date: {e['start_date']}\n"
                f"Office: {e['office']}\n"
                f"Manager: {e['manager']}\n"
                f"Clearance Level: [RESTRICTED — request via "
                f"security_audit]\n"
                f"Emergency Contact: [MIGRATED — see historical "
                f"employee records]"
            )
        return f"No employee found for '{name}'."

    def security_audit(self, employee_id: str) -> str:
        eid = employee_id.strip()
        if eid in self.security:
            s = self.security[eid]
            self.security_fetched = eid
            return (
                f"Security Audit Record — {eid} ({s['name']})\n"
                f"Clearance Level: {s['clearance']}\n"
                f"Granted: {s['granted']}\n"
                f"Sponsor: {s['sponsor']}\n"
                f"Last Review: {s['last_review']}\n"
                f"Expires: {s['expires']}\n"
                f"Access Groups: {s['access_groups']}"
            )
        return f"No security audit record for '{employee_id}'."

    def personnel_file(self, employee_id: str) -> str:
        eid = employee_id.strip()
        if eid not in {"E-1847", "E-2234"}:
            return f"No personnel file for '{employee_id}'."
        return (
            f"Personnel File — {eid}\n"
            f"Contains: compensation history, performance reviews, "
            f"PTO balance.\n"
            f"Note: Emergency contact information is not stored in "
            f"personnel files.\n"
            f"It was moved to HR onboarding records during the 2020 "
            f"system migration."
        )

    def onboarding_records(self, employee_id: str) -> str:
        eid = employee_id.strip()
        if eid in self.onboarding:
            o = self.onboarding[eid]
            self.onboarding_fetched = eid
            return (
                f"HR Onboarding Record — {eid} ({o['name']})\n"
                f"Original Onboarding Date: {o['onboarding_date']}\n"
                f"Emergency Contact: {o['emergency_contact']}\n"
                f"Dietary Restrictions: {o['dietary']}\n"
                f"T-Shirt Size: {o['tshirt']}\n"
                f"Equipment Issued: {o['equipment']}"
            )
        return f"No onboarding record for '{employee_id}'."

    def compliance_check(self, employee_id: str) -> str:
        return (
            f"Compliance Status — {employee_id}: All mandatory trainings complete.\n"
            f"Last security awareness training: 2024-11-15.\n"
            f"No outstanding compliance items."
        )

    def hr_directory(self, name: str) -> str:
        key = name.strip().lower()
        if key in self.employees:
            e = self.employees[key]
            return (
                f"HR Directory Entry — {name.title()}\n"
                f"Department: {e['department']} | "
                f"Reports to: {e['manager']}\n"
                f"Office: {e['office']} | Extension: {e['extension']}\n"
                f"Status: Active | Full-time"
            )
        return f"No directory entry for '{name}'."

    def submit_report(self, employee_name: str, report: str) -> str:
        return f"Report submitted for {employee_name}."


def _validate_data_gap_recovery_stateful(args: dict[str, Any]) -> bool:
    text = f"{args.get('employee_name', '')} {args.get('report', '')}".lower()
    has_employee = "engineering" in text
    has_clearance = "l3" in text or "confidential" in text
    has_contact = "michael" in text and ("867-5309" in text or "spouse" in text)
    return has_employee and has_clearance and has_contact


def _build_data_gap_recovery_stateful() -> tuple[Workflow, callable]:
    db = HRRecordsSystem()
    tools: dict[str, ToolDef] = {
        "get_employee": ToolDef(
            spec=ToolSpec(
                name="get_employee",
                description="Look up an employee record by name.",
                parameters=EmployeeNameParams,
            ),
            callable=lambda **kw: db.get_employee(kw["name"]),
        ),
        "security_audit": ToolDef(
            spec=ToolSpec(
                name="security_audit",
                description="Query security audit records for an employee.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.security_audit(kw["employee_id"]),
        ),
        "personnel_file": ToolDef(
            spec=ToolSpec(
                name="personnel_file",
                description="Access an employee's personnel file.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.personnel_file(kw["employee_id"]),
        ),
        "onboarding_records": ToolDef(
            spec=ToolSpec(
                name="onboarding_records",
                description="Access HR onboarding records for an employee.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.onboarding_records(kw["employee_id"]),
        ),
        "compliance_check": ToolDef(
            spec=ToolSpec(
                name="compliance_check",
                description="Check compliance status for an employee.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.compliance_check(kw["employee_id"]),
        ),
        "hr_directory": ToolDef(
            spec=ToolSpec(
                name="hr_directory",
                description="Look up an employee in the HR directory.",
                parameters=EmployeeNameParams,
            ),
            callable=lambda **kw: db.hr_directory(kw["name"]),
        ),
        "submit_report": ToolDef(
            spec=ToolSpec(
                name="submit_report",
                description="Submit a completed employee profile report.",
                parameters=SubmitReportParams,
            ),
            callable=lambda **kw: db.submit_report(kw.get("employee_name", ""), kw.get("report", "")),
        ),
    }
    workflow = Workflow(
        name="data_gap_recovery_stateful",
        description="Build a complete employee profile by resolving missing data across multiple systems",
        tools=tools,
        required_steps=["get_employee"],
        terminal_tool="submit_report",
        system_prompt_template=(
            "You are an HR systems assistant. Use the available tools to "
            "gather the requested employee information and submit a complete "
            "profile report."
        ),
    )
    validate_state = lambda: (
        db.employee_looked_up is not None
        and db.security_fetched == "E-1847"
        and db.onboarding_fetched == "E-1847"
    )
    return workflow, validate_state


data_gap_recovery_stateful = EvalScenario(
    name="data_gap_recovery_stateful",
    description="Stateful data gap recovery — ID threading through breadcrumb hints and dead ends.",
    workflow=_placeholder_workflow("data_gap_recovery_stateful", "submit_report", ["get_employee"]),
    user_message=(
        "Pull together a complete profile for Sarah Chen — we need her "
        "clearance level and emergency contact for the onboarding audit."
    ),
    validate=_validate_data_gap_recovery_stateful,
    build_workflow=_build_data_gap_recovery_stateful,
    tags=["stateful", "model_quality", "reasoning"],
    ideal_iterations=5,
)

