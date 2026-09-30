"""Brick's spatial capability selection rule, with the cost term replaced by latency.

Source: "Brick: Spatial Capability Routing for the MoM Paradigm", arXiv 2606.13241
(https://arxiv.org/html/2606.13241), sections 7.3 to 7.6, Table 6, and the reference router
in https://github.com/regolo-ai/brick-SR1 (commit 950f2808), file
`apps/router/src/spatial-router/pkg/brickrouting/router.go`. Its `scoreModelsWithConfig`,
`tauQueryFrom` and `effectiveParams` functions are what this module ports.

For a request x with capability distribution p(x) over the six capability dimensions and a
complexity label with confidence c:

    tau_q = c * tau[label] + (1 - c) * tau[medium]          tau = {easy .55, medium .72, hard .88}
    z_q   = b + mu * logit(clip(tau_q))                      required level, in logit space
    r_c   = p_c * z_q                                        requirement per dimension
    v_mc  = p_c * logit(clip(s_mc, .02, .98))                capacity of arm m
    D_m   = sqrt( sum_c max(0, r_c - v_mc)^2 + lam * sum_c max(0, v_mc - r_c)^2 )
    J_m   = D_m + beta * a_m                                 pick argmin J

Brick's a_m is a cost weight: the model's output price divided by the maximum price over the
eligible pool (`economics/pricing.go` DynamicCostWeight). Here a_m is the expected end-to-end
latency divided by the maximum over the candidate arms, so it lies in (0, 1] on the same
scale (see lucerouter/latency.py).

Ties: arms whose J is within `tie_eps` (0.03) of the best J are compared on expected success
sum_c p_c * s_mc, and the higher one wins. If that also ties, the lower latency wins. Brick
breaks this second tie on the lower static cost_weight + latency_weight.

The knob r in [-1, 1] moves mu, b, beta and lambda together (paper section 7.6, Table 6):

    p+ = max(r, 0)^alpha,  p- = max(-r, 0)^alpha
    mu = mu0 * exp(p+ ln A+mu + p- ln A-mu)       b   = b0 + p+ A+b + p- A-b
    beta = beta0 * exp(-p+ ln A+beta + p- ln A-beta)
    lam  = lam0  * exp(-p+ ln A+lam  + p- ln A-lam)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Routing order (alphabetical, Brick's `capabilityLabelOrder`).
CAPABILITIES = ("coding", "creative_synthesis", "instruction_following",
                "math_reasoning", "planning_agentic", "world_knowledge")

TAU = {"easy": 0.55, "medium": 0.72, "hard": 0.88}
SKILL_CLIP = (0.02, 0.98)
TAU_CLIP = (0.02, 0.98)
TIE_EPS = 0.03

# Paper Table 6 / brick-SR1 config.yaml `skill_router.math` (locked constants).
MU0, B0, BETA0, LAM0 = 0.345170, 0.822235, 0.230778, 0.045207
ALPHA = 2.920351
A_PLUS = {"mu": 13.034935, "b": 5.294173, "beta": 6559.073066, "lam": 49.54794}
A_MINUS = {"mu": 0.081493, "b": -1.349259, "beta": 8.834043, "lam": 1002.068256}


@dataclass(frozen=True)
class BrickParams:
    mu: float
    b: float
    beta: float
    lam: float


def effective_params(r: float = 0.0) -> BrickParams:
    """Brick's `effectiveParams`: the routing preference r in [-1, 1] to (mu, b, beta, lam)."""
    r = max(-1.0, min(1.0, float(r)))
    pp = max(r, 0.0) ** ALPHA
    pm = max(-r, 0.0) ** ALPHA
    return BrickParams(
        mu=MU0 * math.exp(pp * math.log(A_PLUS["mu"]) + pm * math.log(A_MINUS["mu"])),
        b=B0 + pp * A_PLUS["b"] + pm * A_MINUS["b"],
        beta=BETA0 * math.exp(-pp * math.log(A_PLUS["beta"]) + pm * math.log(A_MINUS["beta"])),
        lam=LAM0 * math.exp(-pp * math.log(A_PLUS["lam"]) + pm * math.log(A_MINUS["lam"])),
    )


def logit(x: float, lo: float, hi: float) -> float:
    x = min(max(x, lo), hi)
    return math.log(x / (1.0 - x))


def normalise_probs(p: dict[str, float]) -> dict[str, float]:
    """L1-renormalise over CAPABILITIES; NaN or negative values become 0, all-zero becomes uniform."""
    vals = {c: float(p.get(c, 0.0)) for c in CAPABILITIES}
    vals = {c: v if v > 0 and not math.isnan(v) else 0.0 for c, v in vals.items()}
    total = sum(vals.values())
    if total <= 0:
        return {c: 1.0 / len(CAPABILITIES) for c in CAPABILITIES}
    return {c: v / total for c, v in vals.items()}


def complexity_label(probs: dict[str, float] | None) -> tuple[str, float]:
    """(argmax label, its probability) from easy/medium/hard probs; ("medium", 1.0) when missing.

    Brick's complexity confidence is the softmax over the easy/medium/hard token logprobs of
    the first generated position, which is what signals.parse_brick_probs returns.
    """
    if not probs:
        return "medium", 1.0
    label = max(TAU, key=lambda k: probs.get(k, 0.0))
    return label, float(probs.get(label, 0.0))


def tau_query(label: str, confidence: float) -> float:
    c = max(0.0, min(1.0, confidence))
    if label not in TAU:
        label, c = "medium", 0.0
    return c * TAU[label] + (1.0 - c) * TAU["medium"]


def required_level(tau_q: float, params: BrickParams) -> float:
    return params.b + params.mu * logit(tau_q, *TAU_CLIP)


def distance(p: dict[str, float], skills: dict[str, float], z_q: float, lam: float) -> float:
    """Asymmetric distance D_m: under-capacity weighted 1, over-capacity weighted lam."""
    under = over = 0.0
    for c in CAPABILITIES:
        r = p[c] * z_q
        v = p[c] * logit(skills[c], *SKILL_CLIP)
        under += max(0.0, r - v) ** 2
        over += max(0.0, v - r) ** 2
    return math.sqrt(under + lam * over)


def expected_success(p: dict[str, float], skills: dict[str, float]) -> float:
    return sum(p[c] * skills[c] for c in CAPABILITIES)


@dataclass
class ArmScore:
    arm: str
    D: float
    lat_ms: float
    a: float
    J: float
    success: float

    def to_dict(self) -> dict:
        return {"D": round(self.D, 4), "lat_ms": round(self.lat_ms, 1), "a": round(self.a, 4),
                "J": round(self.J, 4), "success": round(self.success, 4)}


def select(p: dict[str, float], skills: dict[str, dict[str, float]], lat_ms: dict[str, float],
           z_q: float, params: BrickParams, tie_eps: float = TIE_EPS) -> tuple[str, list[ArmScore], float]:
    """(chosen arm, scores sorted by J, margin). ``skills`` and ``lat_ms`` cover the candidates.

    margin = J of the best other arm minus J of the chosen arm (0 when the tie band decided).
    """
    arms = list(skills)
    lat_max = max(lat_ms[m] for m in arms) or 1.0
    scores = []
    for m in arms:
        d = distance(p, skills[m], z_q, params.lam)
        a = lat_ms[m] / lat_max
        scores.append(ArmScore(m, d, lat_ms[m], a, d + params.beta * a, expected_success(p, skills[m])))
    scores.sort(key=lambda s: s.J)
    band = [s for s in scores if s is scores[0] or s.J - scores[0].J < tie_eps]
    best = max(band, key=lambda s: (s.success, -s.lat_ms))
    others = [s.J for s in scores if s is not best]
    margin = max(0.0, min(others) - best.J) if others else 1.0
    return best.arm, scores, margin
