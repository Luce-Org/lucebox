"""Compaction chain scenarios — 10-step medical investigation dependency chain.

Four scenarios (baseline + P1/P2/P3) share one backend.  Each tool returns
~500-800 chars of realistic medical detail with one key fact and an ID needed
as input to the next call.  The model must thread IDs through the chain AND
cite specific values from early steps in the final treatment plan.

Dependency chain:
  patient_lookup → pull_records → order_labs → review_imaging →
  request_referral → check_pharmacy → verify_insurance →
  request_prior_auth → schedule_appointment → submit_treatment_plan
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .._forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario, _placeholder_workflow


# ── Pydantic parameter models ────────────────────────────────


class PatientLookupParams(BaseModel):
    patient_name: str = Field(description="Full name of the patient to look up")


class PullRecordsParams(BaseModel):
    mrn: str = Field(description="Medical Record Number from patient_lookup")


class OrderLabsParams(BaseModel):
    encounter_id: str = Field(description="Encounter ID from pull_records")


class ReviewImagingParams(BaseModel):
    lab_id: str = Field(description="Lab order ID from order_labs")


class RequestReferralParams(BaseModel):
    imaging_id: str = Field(description="Imaging study ID from review_imaging")


class CheckPharmacyParams(BaseModel):
    referral_id: str = Field(description="Referral ID from request_referral")


class VerifyInsuranceParams(BaseModel):
    patient_mrn: str = Field(description="Patient MRN from patient_lookup")


class RequestPriorAuthParams(BaseModel):
    plan_id: str = Field(description="Insurance plan ID from verify_insurance")
    referral_id: str = Field(description="Referral ID from request_referral")


class ScheduleAppointmentParams(BaseModel):
    auth_id: str = Field(description="Prior authorization ID from request_prior_auth")
    referral_id: str = Field(description="Referral ID from request_referral")


class TreatmentPlanParams(BaseModel):
    summary: str = Field(
        description=(
            "Treatment plan summary citing patient info, diagnosis, key lab values, "
            "imaging findings, specialist, medication concerns, and scheduled appointment"
        ),
    )


# ── Tool result strings ──────────────────────────────────────
#
# Each result is ~500-800 chars of realistic medical detail.  The key fact
# and threading ID are embedded in the noise — compaction must preserve
# enough context for the model to cite them in the final treatment plan.

_PATIENT_RESULT = (
    "═══ PATIENT RECORD ═══\n"
    "Name: Margaret Chen | MRN: MRN-84201 | DOB: 1958-04-12 (age 67)\n"
    "Sex: Female | Blood Type: A+ | Language: English/Mandarin\n"
    "Address: 1847 Oakwood Dr, Unit 3B, Riverside CA 92501\n"
    "Phone: (951) 555-0147 | Emergency: David Chen (spouse) (951) 555-0148\n"
    "Insurance: BlueCross PPO (see verify_insurance for details)\n\n"
    "Active Conditions:\n"
    "  • Type 2 Diabetes Mellitus (E11.9) — dx 2018, managed w/ oral agents\n"
    "  • Essential Hypertension (I10) — dx 2015, controlled on ACE inhibitor\n"
    "  • Hyperlipidemia (E78.5) — dx 2019, on statin therapy\n"
    "  • Obesity, BMI 31.4 (E66.01)\n\n"
    "Allergies: Sulfonamides (rash), Iodine contrast (mild urticaria)\n"
    "PCP: Dr. James Robbins, Riverside Family Medicine\n"
    "Last Visit: 2024-11-15 (routine diabetes follow-up, stable at that time)\n"
    "Pharmacy: CVS #4218, 2900 University Ave, Riverside CA"
)

_ENCOUNTER_RESULT = (
    "═══ ENCOUNTER RECORD ═══\n"
    "Encounter ID: ENC-20250305 | Date: 2025-03-05 | Type: Office Visit\n"
    "Provider: Dr. James Robbins | Facility: Riverside Family Medicine\n\n"
    "Chief Complaint: Persistent fatigue and increased urination x 3 weeks.\n"
    "Patient reports waking 3-4x/night to urinate (nocturia). Daytime fatigue\n"
    "interfering with daily activities. Unintentional weight loss ~4 lbs over\n"
    "past month. Increased thirst. Denies fever, dysuria, hematuria.\n\n"
    "Vitals:\n"
    "  BP: 142/88 mmHg | HR: 78 bpm | Temp: 98.4°F | SpO2: 97%\n"
    "  Weight: 187 lbs (prev 191 lbs on 2024-11-15) | BMI: 31.4\n\n"
    "Point-of-Care Testing:\n"
    "  Finger-stick glucose: 248 mg/dL (non-fasting, 2hrs post-meal)\n"
    "  Urine dipstick: glucose 3+, protein 1+, ketones negative\n\n"
    "Assessment: Suspected uncontrolled diabetes with possible renal involvement\n"
    "given proteinuria. HbA1c and comprehensive metabolic panel ordered.\n"
    "Plan: Stat labs, renal ultrasound if eGFR reduced. Follow up in 1 week."
)

_LABS_RESULT = (
    "═══ LABORATORY RESULTS ═══\n"
    "Lab Order: LAB-7718 | Collected: 2025-03-05 09:15 | Resulted: 2025-03-05 14:30\n"
    "Ordering Provider: Dr. James Robbins | Patient: Margaret Chen (MRN-84201)\n\n"
    "Comprehensive Metabolic Panel:\n"
    "  Glucose (fasting): 212 mg/dL  [H]  (ref: 70-100)\n"
    "  BUN: 28 mg/dL  [H]  (ref: 7-20)\n"
    "  Creatinine: 1.6 mg/dL  [H]  (ref: 0.6-1.2)\n"
    "  eGFR: 48 mL/min/1.73m²  [L]  (ref: >60)\n"
    "  Sodium: 139 mEq/L  (ref: 136-145)\n"
    "  Potassium: 4.8 mEq/L  (ref: 3.5-5.0)\n"
    "  Calcium: 9.2 mg/dL  (ref: 8.5-10.5)\n"
    "  Albumin: 3.4 g/dL  [L]  (ref: 3.5-5.0)\n\n"
    "Hemoglobin A1c:\n"
    "  HbA1c: 9.2%  [H]  (ref: <5.7% normal, <7.0% diabetic target)\n"
    "  Estimated Average Glucose: 217 mg/dL\n\n"
    "Urinalysis:\n"
    "  Urine Albumin/Creatinine Ratio (UACR): 182 mg/g  [H]  (ref: <30)\n"
    "  Category: A3 (severely increased albuminuria)\n\n"
    "Interpretation: HbA1c 9.2% significantly above target — diabetes poorly\n"
    "controlled. eGFR 48 classifies as Stage 3a CKD. UACR 182 indicates\n"
    "significant albuminuria. Pattern consistent with diabetic nephropathy.\n"
    "Recommend nephrology referral and renal imaging."
)

_IMAGING_RESULT = (
    "═══ IMAGING REPORT ═══\n"
    "Study: IMG-3304 | Modality: Renal Ultrasound (bilateral)\n"
    "Date: 2025-03-06 | Performed by: Sarah Kim, RDMS\n"
    "Interpreted by: Dr. Michael Torres, Radiology\n"
    "Patient: Margaret Chen (MRN-84201) | Indication: Elevated creatinine, low eGFR\n\n"
    "Findings:\n"
    "  Right Kidney: 10.2 cm length (normal 9-13 cm). Mild cortical thinning\n"
    "    with increased echogenicity. No hydronephrosis. No calculi or masses.\n"
    "    Resistive index 0.74 (mildly elevated, ref <0.70).\n"
    "  Left Kidney: 10.0 cm length. Similar mild cortical thinning and\n"
    "    increased echogenicity. No hydronephrosis. No calculi or masses.\n"
    "    Resistive index 0.72.\n"
    "  Bladder: Normal wall thickness, no post-void residual measured.\n\n"
    "Impression:\n"
    "  1. Bilateral mild cortical thinning with increased echogenicity,\n"
    "     changes consistent with early diabetic nephropathy (medical renal disease).\n"
    "  2. Mildly elevated resistive indices bilaterally, suggesting early\n"
    "     intrarenal vascular changes.\n"
    "  3. No obstructive uropathy, calculi, or masses identified.\n\n"
    "Recommend: Correlate with nephrology consultation for CKD management."
)

_REFERRAL_RESULT = (
    "═══ SPECIALIST REFERRAL ═══\n"
    "Referral ID: REF-5521 | Created: 2025-03-06 | Status: Submitted\n"
    "From: Dr. James Robbins, Riverside Family Medicine\n"
    "To: Dr. Anita Patel, Nephrology\n"
    "Facility: Metro Kidney Associates, 3200 Central Ave, Riverside CA 92506\n"
    "Phone: (951) 555-0290 | Fax: (951) 555-0291\n\n"
    "Patient: Margaret Chen (MRN-84201) | DOB: 1958-04-12\n\n"
    "Reason for Referral:\n"
    "  Stage 3a CKD (eGFR 48) with diabetic nephropathy confirmed on renal\n"
    "  ultrasound. UACR 182 mg/g (A3 category). HbA1c 9.2% — poorly controlled\n"
    "  diabetes driving renal decline. Needs specialist management for:\n"
    "    • CKD staging and progression monitoring\n"
    "    • Renoprotective medication optimization\n"
    "    • Blood pressure target adjustment (currently 142/88)\n"
    "    • Diabetes medication review (metformin dose vs. renal function)\n\n"
    "Enclosed: Lab results (LAB-7718), imaging report (IMG-3304)\n"
    "Urgency: Routine (within 2 weeks) | Prior auth may be required"
)

_PHARMACY_RESULT = (
    "═══ MEDICATION REVIEW ═══\n"
    "Patient: Margaret Chen (MRN-84201) | Pharmacy: CVS #4218\n"
    "Review Date: 2025-03-06 | Reviewed by: PharmD system\n\n"
    "Current Medications:\n"
    "  1. Metformin 1000mg PO BID — diabetes (since 2018)\n"
    "  2. Lisinopril 20mg PO daily — hypertension/renoprotection (since 2015)\n"
    "  3. Atorvastatin 40mg PO QHS — hyperlipidemia (since 2019)\n"
    "  4. Aspirin 81mg PO daily — cardiovascular prophylaxis (since 2019)\n\n"
    "⚠ DRUG-CONDITION INTERACTION ALERT:\n"
    "  Metformin + CKD Stage 3a (eGFR 48 mL/min):\n"
    "  FDA guidance: metformin contraindicated when eGFR falls below 30.\n"
    "  Dose reduction recommended at eGFR 30-45. Patient currently at 48,\n"
    "  approaching the threshold for mandatory dose reduction. With progressive\n"
    "  diabetic nephropathy, eGFR may decline further.\n\n"
    "  Recommendation: Consider switching to SGLT2 inhibitor (empagliflozin\n"
    "  10mg daily). Evidence: EMPA-KIDNEY trial showed 28% reduction in CKD\n"
    "  progression. Dual benefit — glycemic control + renoprotection.\n"
    "  Discuss with nephrology at upcoming referral.\n\n"
    "  Lisinopril: Continue — ACE inhibitors are first-line for diabetic CKD.\n"
    "  Monitor potassium (currently 4.8, near upper limit).\n\n"
    "No other interactions identified. Adherence records: good (90%+ refill rate)."
)

_INSURANCE_RESULT = (
    "═══ INSURANCE VERIFICATION ═══\n"
    "Patient: Margaret Chen (MRN-84201)\n"
    "Verification Date: 2025-03-06 | Verified by: Auto-eligibility system\n\n"
    "Plan: PLAN-BC-4490 | BlueCross Blue Shield PPO\n"
    "Group: GRP-METRO-0088 (Metro Riverside Employers Group)\n"
    "Member ID: BCB-449017823 | Effective: 2024-01-01 | Status: Active\n\n"
    "Coverage Details:\n"
    "  Primary Care Copay: $25 | Specialist Copay: $40\n"
    "  Deductible: $1,500 individual ($842 met YTD)\n"
    "  Out-of-Pocket Max: $6,000 individual\n"
    "  Lab/Imaging: Covered at 80% after deductible\n"
    "  Prescriptions: Tier 1 $10 / Tier 2 $35 / Tier 3 $60\n\n"
    "Authorization Requirements:\n"
    "  ✓ Nephrology office visit — PRIOR AUTH REQUIRED\n"
    "  ✓ Renal ultrasound — pre-authorized (already completed)\n"
    "  ✗ Renal biopsy — would require separate auth if needed\n\n"
    "Network Status:\n"
    "  Dr. Anita Patel / Metro Kidney Associates — IN-NETWORK\n"
    "  Estimated patient responsibility: $40 copay per visit\n\n"
    "Action Required: Submit prior authorization for nephrology referral\n"
    "before scheduling. Use referral REF-5521 and plan PLAN-BC-4490."
)

_PRIOR_AUTH_RESULT = (
    "═══ PRIOR AUTHORIZATION ═══\n"
    "Auth ID: AUTH-9917 | Status: APPROVED\n"
    "Submitted: 2025-03-06 14:22 | Decided: 2025-03-06 14:23 (auto-approved)\n"
    "Plan: PLAN-BC-4490 (BlueCross PPO) | Referral: REF-5521\n\n"
    "Patient: Margaret Chen (MRN-84201)\n"
    "Requesting Provider: Dr. James Robbins\n"
    "Servicing Provider: Dr. Anita Patel, Metro Kidney Associates\n\n"
    "Authorized Services:\n"
    "  • Nephrology consultation (CPT 99245) — up to 3 visits\n"
    "  • Follow-up visits (CPT 99214) — included in 3-visit authorization\n"
    "  • Additional labs as ordered by specialist — covered under plan\n\n"
    "Authorization Window: 90 days from 2025-03-06 (expires 2025-06-04)\n"
    "Visits Authorized: 3 | Visits Used: 0\n\n"
    "Conditions:\n"
    "  - Must use in-network provider listed above\n"
    "  - Extension request required if >3 visits needed\n"
    "  - Procedures beyond office visit (e.g. biopsy) need separate auth\n\n"
    "Next Step: Schedule appointment with Dr. Patel within authorization window."
)

_APPOINTMENT_RESULT = (
    "═══ APPOINTMENT CONFIRMATION ═══\n"
    "Appointment ID: APPT-20250312\n"
    "Date: Wednesday, March 12, 2025 | Time: 2:00 PM\n"
    "Duration: 45 minutes (new patient nephrology consultation)\n\n"
    "Patient: Margaret Chen (MRN-84201)\n"
    "Provider: Dr. Anita Patel, MD, FASN\n"
    "Facility: Metro Kidney Associates\n"
    "  3200 Central Ave, Suite 200, Riverside CA 92506\n"
    "  Phone: (951) 555-0290\n\n"
    "Authorization: AUTH-9917 (verified, 3 visits remaining)\n"
    "Copay: $40 (collect at check-in)\n\n"
    "Pre-Visit Instructions:\n"
    "  □ Bring photo ID and insurance card\n"
    "  □ Bring list of current medications\n"
    "  □ Lab results (LAB-7718) and imaging report (IMG-3304) will be\n"
    "    sent electronically — bring paper copies as backup\n"
    "  □ Fast for 8 hours before appointment (labs may be redrawn)\n"
    "  □ Arrive 15 minutes early for new patient paperwork\n\n"
    "Reminder: Patient will receive automated call 48hrs before appointment\n"
    "and SMS reminder morning-of. Cancellation requires 24hr notice."
)


# ── Backend ──────────────────────────────────────────────────


class MedicalInvestigation:
    """Stateful backend for compaction chain — 10-step medical investigation."""

    def __init__(self) -> None:
        self.calls_in_order: list[str] = []

    def _record(self, tool: str) -> None:
        self.calls_in_order.append(tool)

    # Step 1 — key: MRN-84201
    def patient_lookup(self, patient_name: str) -> str:
        self._record("patient_lookup")
        name = patient_name.strip().lower()
        if "margaret" not in name and "chen" not in name:
            return (
                f"No patient found for '{patient_name}'. "
                "Try: Margaret Chen."
            )
        return _PATIENT_RESULT

    # Step 2 — key: ENC-20250305
    def pull_records(self, mrn: str) -> str:
        self._record("pull_records")
        if mrn.strip().upper() != "MRN-84201":
            return f"No records for MRN '{mrn}'. Expected: MRN-84201."
        return _ENCOUNTER_RESULT

    # Step 3 — key: LAB-7718, HbA1c 9.2%, eGFR 48
    def order_labs(self, encounter_id: str) -> str:
        self._record("order_labs")
        if encounter_id.strip().upper() != "ENC-20250305":
            return f"Unknown encounter '{encounter_id}'. Expected: ENC-20250305."
        return _LABS_RESULT

    # Step 4 — key: IMG-3304, cortical thinning
    def review_imaging(self, lab_id: str) -> str:
        self._record("review_imaging")
        if lab_id.strip().upper() != "LAB-7718":
            return f"Unknown lab order '{lab_id}'. Expected: LAB-7718."
        return _IMAGING_RESULT

    # Step 5 — key: REF-5521, Dr. Patel
    def request_referral(self, imaging_id: str) -> str:
        self._record("request_referral")
        if imaging_id.strip().upper() != "IMG-3304":
            return f"Unknown imaging study '{imaging_id}'. Expected: IMG-3304."
        return _REFERRAL_RESULT

    # Step 6 — key: metformin contraindication, empagliflozin
    def check_pharmacy(self, referral_id: str) -> str:
        self._record("check_pharmacy")
        if referral_id.strip().upper() != "REF-5521":
            return f"Unknown referral '{referral_id}'. Expected: REF-5521."
        return _PHARMACY_RESULT

    # Step 7 — key: PLAN-BC-4490
    def verify_insurance(self, patient_mrn: str) -> str:
        self._record("verify_insurance")
        if patient_mrn.strip().upper() != "MRN-84201":
            return f"No insurance on file for MRN '{patient_mrn}'. Expected: MRN-84201."
        return _INSURANCE_RESULT

    # Step 8 — key: AUTH-9917
    def request_prior_auth(self, plan_id: str, referral_id: str) -> str:
        self._record("request_prior_auth")
        if plan_id.strip().upper() != "PLAN-BC-4490":
            return f"Unknown plan '{plan_id}'. Expected: PLAN-BC-4490."
        if referral_id.strip().upper() != "REF-5521":
            return f"Unknown referral '{referral_id}'. Expected: REF-5521."
        return _PRIOR_AUTH_RESULT

    # Step 9 — key: APPT-20250312
    def schedule_appointment(self, auth_id: str, referral_id: str) -> str:
        self._record("schedule_appointment")
        if auth_id.strip().upper() != "AUTH-9917":
            return f"Unknown authorization '{auth_id}'. Expected: AUTH-9917."
        if referral_id.strip().upper() != "REF-5521":
            return f"Unknown referral '{referral_id}'. Expected: REF-5521."
        return _APPOINTMENT_RESULT

    # Step 10 (terminal)
    def submit_treatment_plan(self, summary: str) -> str:
        self._record("submit_treatment_plan")
        return summary  # echo-back terminal


# ── Expected call order (for validate_state) ─────────────────

_EXPECTED_ORDER = [
    "patient_lookup",
    "pull_records",
    "order_labs",
    "review_imaging",
    "request_referral",
    "check_pharmacy",
    "verify_insurance",
    "request_prior_auth",
    "schedule_appointment",
    "submit_treatment_plan",
]

_REQUIRED_STEPS = _EXPECTED_ORDER[:-1]  # everything except terminal


# ── Workflow builder ─────────────────────────────────────────


def _build_compaction_chain() -> tuple[Workflow, callable]:
    db = MedicalInvestigation()

    tools: dict[str, ToolDef] = {
        "patient_lookup": ToolDef(
            spec=ToolSpec(
                name="patient_lookup",
                description="Look up a patient by name. Returns MRN and demographics.",
                parameters=PatientLookupParams,
            ),
            callable=lambda **kw: db.patient_lookup(kw["patient_name"]),
        ),
        "pull_records": ToolDef(
            spec=ToolSpec(
                name="pull_records",
                description="Pull recent medical records for a patient by MRN.",
                parameters=PullRecordsParams,
            ),
            callable=lambda **kw: db.pull_records(kw["mrn"]),
        ),
        "order_labs": ToolDef(
            spec=ToolSpec(
                name="order_labs",
                description="Order and retrieve lab results for an encounter.",
                parameters=OrderLabsParams,
            ),
            callable=lambda **kw: db.order_labs(kw["encounter_id"]),
        ),
        "review_imaging": ToolDef(
            spec=ToolSpec(
                name="review_imaging",
                description="Review imaging studies ordered alongside lab work.",
                parameters=ReviewImagingParams,
            ),
            callable=lambda **kw: db.review_imaging(kw["lab_id"]),
        ),
        "request_referral": ToolDef(
            spec=ToolSpec(
                name="request_referral",
                description="Request a specialist referral based on imaging findings.",
                parameters=RequestReferralParams,
            ),
            callable=lambda **kw: db.request_referral(kw["imaging_id"]),
        ),
        "check_pharmacy": ToolDef(
            spec=ToolSpec(
                name="check_pharmacy",
                description="Check current medications and flag contraindications for a referral.",
                parameters=CheckPharmacyParams,
            ),
            callable=lambda **kw: db.check_pharmacy(kw["referral_id"]),
        ),
        "verify_insurance": ToolDef(
            spec=ToolSpec(
                name="verify_insurance",
                description="Verify insurance coverage and authorization requirements for a patient.",
                parameters=VerifyInsuranceParams,
            ),
            callable=lambda **kw: db.verify_insurance(kw["patient_mrn"]),
        ),
        "request_prior_auth": ToolDef(
            spec=ToolSpec(
                name="request_prior_auth",
                description="Request prior authorization from the insurance plan for a referral.",
                parameters=RequestPriorAuthParams,
            ),
            callable=lambda **kw: db.request_prior_auth(kw["plan_id"], kw["referral_id"]),
        ),
        "schedule_appointment": ToolDef(
            spec=ToolSpec(
                name="schedule_appointment",
                description="Schedule a specialist appointment using authorization and referral.",
                parameters=ScheduleAppointmentParams,
            ),
            callable=lambda **kw: db.schedule_appointment(kw["auth_id"], kw["referral_id"]),
        ),
        "submit_treatment_plan": ToolDef(
            spec=ToolSpec(
                name="submit_treatment_plan",
                description=(
                    "Submit the final treatment plan summarizing the investigation. "
                    "Must cite patient info, diagnosis, key lab values, imaging findings, "
                    "specialist referral, medication concerns, and appointment details."
                ),
                parameters=TreatmentPlanParams,
            ),
            callable=lambda **kw: db.submit_treatment_plan(kw.get("summary", "")),
        ),
    }

    workflow = Workflow(
        name="compaction_chain",
        description=(
            "10-step medical investigation: patient lookup through treatment plan. "
            "Each step returns an ID needed by the next step."
        ),
        tools=tools,
        required_steps=_REQUIRED_STEPS,
        terminal_tool="submit_treatment_plan",
        system_prompt_template=(
            "You are a medical case coordinator. Investigate the patient and "
            "build a treatment plan by following each step in order. Each tool "
            "returns an ID you will need for the next call.\n\n"
            "Follow this exact sequence:\n"
            "1. patient_lookup — look up the patient by name\n"
            "2. pull_records — use the MRN from step 1\n"
            "3. order_labs — use the encounter ID from step 2\n"
            "4. review_imaging — use the lab order ID from step 3\n"
            "5. request_referral — use the imaging ID from step 4\n"
            "6. check_pharmacy — use the referral ID from step 5\n"
            "7. verify_insurance — use the patient MRN from step 1\n"
            "8. request_prior_auth — use the plan ID from step 7 and referral ID from step 5\n"
            "9. schedule_appointment — use the auth ID from step 8 and referral ID from step 5\n"
            "10. submit_treatment_plan — summarize findings citing key facts from each step"
        ),
    )

    def validate_state() -> bool:
        # All 10 tools called, in correct order
        return db.calls_in_order == _EXPECTED_ORDER

    return workflow, validate_state


# ── Validation ───────────────────────────────────────────────


def _validate_treatment_plan(args: dict[str, Any]) -> bool:
    """Check that the treatment plan cites key facts from across the chain.

    Each checked value only appears in tool results — not in the user message
    or system prompt — so the model can only cite them if it saw (or remembered)
    the tool output.

    Requires all 5 checks to pass:
      1. MRN-84201 (step 1 only)
      2. HbA1c 9.2% or eGFR 48 (step 3 only)
      3. Cortical thinning or nephropathy (step 4 only)
      4. Patel or nephrology (step 5 only)
      5. Metformin or empagliflozin or SGLT2 (step 6 only)
    """
    text = (args.get("summary", "") or "").lower().replace(",", "")
    checks = [
        # Step 1: patient MRN (not in user message)
        "mrn-84201" in text or "84201" in text,
        # Step 3: lab values (only from tool result)
        any(t in text for t in ["9.2", "hba1c", "egfr 48", "egfr: 48", "48 ml"]),
        # Step 4: imaging findings (only from tool result)
        any(t in text for t in ["cortical thinning", "nephropathy"]),
        # Step 5: specialist (only from tool result)
        any(t in text for t in ["patel", "nephrology"]),
        # Step 6: medication concern (only from tool result)
        any(t in text for t in ["metformin", "sglt2", "empagliflozin"]),
    ]
    return all(checks)


# ── Three scenarios, three budgets ───────────────────────────

_TERMINAL = "submit_treatment_plan"
_USER_MESSAGE = (
    "Patient Margaret Chen called about persistent fatigue. Investigate her case "
    "fully — look her up, pull records, run labs, review imaging, get a specialist "
    "referral, check medications, verify insurance, obtain authorization, schedule "
    "the appointment, and submit a treatment plan summarizing everything."
)

compaction_chain_baseline = EvalScenario(
    name="compaction_chain_baseline",
    description=(
        "10-step medical dependency chain with full budget (no compaction). "
        "Baseline for measuring degradation under P1/P2/P3."
    ),
    workflow=_placeholder_workflow(
        "compaction_chain_baseline", _TERMINAL, _REQUIRED_STEPS,
    ),
    user_message=_USER_MESSAGE,
    max_iterations=20,
    validate=_validate_treatment_plan,
    build_workflow=_build_compaction_chain,
    tags=["stateful"],
    ideal_iterations=10,
)

compaction_chain_p1 = EvalScenario(
    name="compaction_chain_p1",
    description=(
        "10-step medical dependency chain under P1 compaction. "
        "Truncated results, but short IDs survive. High correctness expected."
    ),
    workflow=_placeholder_workflow(
        "compaction_chain_p1", _TERMINAL, _REQUIRED_STEPS,
    ),
    user_message=_USER_MESSAGE,
    budget_tokens=3600,
    max_iterations=20,
    validate=_validate_treatment_plan,
    build_workflow=_build_compaction_chain,
    tags=["stateful", "compaction"],
    ideal_iterations=10,
)

compaction_chain_p2 = EvalScenario(
    name="compaction_chain_p2",
    description=(
        "10-step medical dependency chain under P2 compaction. "
        "Tool results dropped mid-chain. Degraded but recoverable from tool_call args."
    ),
    workflow=_placeholder_workflow(
        "compaction_chain_p2", _TERMINAL, _REQUIRED_STEPS,
    ),
    user_message=_USER_MESSAGE,
    budget_tokens=2200,
    max_iterations=20,
    validate=_validate_treatment_plan,
    build_workflow=_build_compaction_chain,
    tags=["stateful", "compaction"],
    ideal_iterations=10,
)

compaction_chain_p3 = EvalScenario(
    name="compaction_chain_p3",
    description=(
        "10-step medical dependency chain under P3 compaction. "
        "Everything dropped except recent window. Chain broken — ~0% expected."
    ),
    workflow=_placeholder_workflow(
        "compaction_chain_p3", _TERMINAL, _REQUIRED_STEPS,
    ),
    user_message=_USER_MESSAGE,
    budget_tokens=1536,
    max_iterations=20,
    validate=_validate_treatment_plan,
    build_workflow=_build_compaction_chain,
    tags=["stateful", "compaction"],
    ideal_iterations=10,
)
