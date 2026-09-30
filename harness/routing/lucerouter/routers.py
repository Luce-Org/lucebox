"""Pluggable routers.

Spec grammar (string form):

    spec    := layer ( ">" layer )*          # wrappers first, one leaf router last
    layer   := name [ ":" param ( "," param )* ]
    param   := key "=" value                 # lists use "|", e.g. models=a|b

e.g. "load_aware:margin_down=0.1>brick:service=brick-max,t_small=0.8"

JSON form: {"router": "load_aware", "params": {...}, "inner": {"router": "brick", ...}}
or a list of {"router", "params"} objects in the same outer→inner order.

Offline use (eval scripts):

    router = build("brick:service=brick-max", load_config(path))
    decision = await router.route(RouteRequest(messages=[...]))
    await router.aclose()

Every router's `decide()` returns a RouteDecision; `route()` adds timing and
the outermost label. Routers may return a model that is not in the config
(e.g. "swarm") as an escalation target; the gateway resolves those via its
`extra_targets` hook.

Routers choose *arms* (config.arms): a backend model plus request overrides such as
thinking on/off. Without an `arms` section every model is one arm, named after it.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx

from . import brick_math, signals
from .capability import DEFAULT_MODEL_DIR, get_classifier, routing_text, run_capability
from .config import Config
from .latency import LatencyProfile, expected_latency_ms, queue_wait_ms
from .types import RouteDecision, RouteRequest

ROUTERS: dict[str, type["Router"]] = {}


def register(cls):
    ROUTERS[cls.name] = cls
    return cls


class Resources:
    """Things routers share: one HTTP client, one cached load monitor (per backend model) and
    the capability classifier function (``capability_fn(text) -> probs``; None = load the
    local ModernBERT lazily)."""

    def __init__(self, config: Config, client: httpx.AsyncClient | None = None, load_ttl_s: float = 0.2,
                 capability_fn=None):
        self.config = config
        self.client = client or httpx.AsyncClient()
        self._own_client = client is None
        self.load = signals.LoadMonitor(config.models, self.client, ttl_s=load_ttl_s)
        self.capability_fn = capability_fn

    async def aclose(self):
        if self._own_client:
            await self.client.aclose()


class Router:
    name = "base"
    wrapper = False

    def __init__(self, config: Config, params: dict, res: Resources, inner: "Router | None" = None):
        self.config = config
        self.params = params
        self.res = res
        self.inner = inner
        self.label = self.name

    # param helpers (values arrive as strings from the spec grammar or typed from JSON)
    def p_str(self, key, default=None):
        v = self.params.get(key, default)
        return None if v in (None, "none", "") else str(v)

    def p_float(self, key, default=None):
        v = self.params.get(key, default)
        return None if v in (None, "none", "") else float(v)

    def p_int(self, key, default=None):
        v = self.p_float(key, default)
        return None if v is None else int(v)

    def p_bool(self, key, default=False):
        v = self.params.get(key, default)
        if isinstance(v, str):
            return v.lower() in ("1", "true", "on", "yes")
        return bool(v)

    def p_list(self, key, default=None):
        v = self.params.get(key, default)
        if v in (None, "none", ""):
            return None
        return list(v) if isinstance(v, (list, tuple)) else [s for s in str(v).split("|") if s]

    def check_arm(self, arm: str | None, key: str):
        if arm is not None and arm not in self.config.arms:
            raise ValueError(f"{self.name}: {key}={arm!r} is not a configured arm (have {sorted(self.config.arms)})")

    async def route(self, req: RouteRequest) -> RouteDecision:
        t0 = time.perf_counter()
        d = await self.decide(req)
        d.latency_ms = (time.perf_counter() - t0) * 1000
        d.router = self.label
        return d

    async def decide(self, req: RouteRequest) -> RouteDecision:
        raise NotImplementedError

    def decision(self, model, reason, scores=None, signals_=None) -> RouteDecision:
        return RouteDecision(model=model, router=self.label, reason=reason,
                             scores=scores or {}, signals=signals_ or {})

    async def aclose(self):
        await self.res.aclose()


# ─── leaf routers ────────────────────────────────────────────────────────

@register
class FixedRouter(Router):
    """fixed:model=X — always arm X (baselines)."""
    name = "fixed"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.model = self.p_str("model")
        if not self.model:
            raise ValueError("fixed router needs model=")

    async def decide(self, req):
        return self.decision(self.model, "fixed")


def _tiers(names: list[str]) -> tuple[str | None, str | None, str]:
    """(small, mid, large) from arms ordered cheapest first."""
    if len(names) >= 3:
        return names[0], names[1], names[-1]
    if len(names) == 2:
        return None, names[0], names[1]
    return None, None, names[0]


@register
class BrickRouter(Router):
    """brick: Brick difficulty classifier + thresholds.

    p_easy >= t_small                             -> small
    p_easy + p_medium >= t_mid and p_hard < h_max -> mid
    otherwise                                     -> large
    small/mid/large are arms, by rank: with the default arms easy -> qwen35-2b,
    medium -> qwen38-27b, hard -> qwen38-27b-think.
    Params: service, t_small, t_mid, h_max, small, mid, large (small/mid may be
    "none"), no_think (llama-server kind only: append the empty think block),
    on_error (arm used when the classifier fails; default large).
    signals.margin = distance of the probs from flipping the decision.
    """
    name = "brick"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        small, mid, large = _tiers([a.name for a in self.config.arms_by_rank()])
        self.small = self.p_str("small", small)
        self.mid = self.p_str("mid", mid)
        self.large = self.p_str("large", large)
        self.t_small = self.p_float("t_small", 0.8)
        self.t_mid = self.p_float("t_mid", 0.5)
        self.h_max = self.p_float("h_max", 0.5)
        self.on_error = self.p_str("on_error", self.large)
        self.service = self.config.service(self.p_str("service", "brick-max"))
        self.suffix = signals.THINK_SUFFIX if self.p_bool("no_think") else self.service.prompt_suffix
        for key in ("small", "mid", "large"):
            self.check_arm(getattr(self, key), key)

    def choose(self, probs: dict) -> tuple[str, str, float]:
        """(model, reason, margin) — pure threshold logic, unit-tested directly."""
        pe, pm, ph = probs["easy"], probs["medium"], probs["hard"]
        if self.small and pe >= self.t_small:
            return self.small, f"p_easy {pe:.2f} >= t_small {self.t_small}", pe - self.t_small
        to_small = self.t_small - pe if self.small else 1.0
        mid_ok = (pe + pm) >= self.t_mid and ph < self.h_max
        if self.mid and mid_ok:
            margin = min((pe + pm) - self.t_mid, self.h_max - ph, to_small)
            return self.mid, f"p_easy+p_medium {pe + pm:.2f} >= t_mid {self.t_mid}, p_hard {ph:.2f} < h_max", margin
        to_mid = max(0.0, self.t_mid - (pe + pm), ph - self.h_max) if self.mid else 1.0
        return self.large, f"p_hard {ph:.2f}, p_easy+p_medium {pe + pm:.2f} below thresholds", min(to_mid, to_small)

    async def decide(self, req):
        text = req.last_user_text()
        t0 = time.perf_counter()
        try:
            probs = await signals.brick_probs(
                self.service.base_url, text, kind=self.service.kind, model=self.service.served_name,
                prompt_suffix=self.suffix, client=self.res.client)
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
            return self.decision(self.on_error, f"brick error: {type(e).__name__}: {e}",
                                 signals_={"error": str(e)})
        brick_ms = (time.perf_counter() - t0) * 1000
        model, reason, margin = self.choose(probs)
        return self.decision(model, reason, scores=probs,
                             signals_={"margin": round(margin, 4), "brick_ms": round(brick_ms, 2),
                                       "service": self.service.name})


def load_skill_table(path: str | Path) -> dict[str, dict[str, float]]:
    """{arm: {capability: s}} from a skill table JSON (eval/fit_skills.py)."""
    data = json.loads(Path(path).read_text())
    table = {}
    for arm, vec in data["arms"].items():
        if isinstance(vec, list):
            vec = dict(zip(data.get("capabilities", brick_math.CAPABILITIES), vec, strict=True))
        missing = set(brick_math.CAPABILITIES) - set(vec)
        if missing:
            raise ValueError(f"skill table {path}: arm {arm!r} lacks {sorted(missing)}")
        table[arm] = {c: float(vec[c]) for c in brick_math.CAPABILITIES}
    return table


@register
class BrickSkillRouter(Router):
    """brick_skill: Brick's spatial capability rule with latency in place of cost.

    J_m = D_m + beta * Lat_m / max_k Lat_k over the candidate arms, argmin J (see brick_math).
    D_m needs the capability probs p(x) (ModernBERT, in-process) and the complexity label and
    confidence (the Brick complexity service, as in `brick`). Lat_m comes from the latency
    profile plus the live queue of the arm's backend (see lucerouter/latency.py).

    Params: service (complexity, default brick-max), skills (skill table JSON, required),
    latency (latency profile JSON; config latency_defaults fill gaps), r (Brick knob in
    [-1, 1], default 0, sets mu/b/beta/lam), beta and lam (override r's values), tie_eps
    (0.03), arms (candidates, default all arms), live_load (default on), capability (model
    dir), max_length (512), no_think (llama-server complexity service only).
    When the capability classifier fails, the complexity label picks the arm like Brick's
    model_map fallback (easy -> cheapest, medium -> middle, hard -> top). A complexity
    failure counts as ("medium", confidence 1), as in Brick.
    """
    name = "brick_skill"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.service = self.config.service(self.p_str("service", "brick-max"))
        self.suffix = signals.THINK_SUFFIX if self.p_bool("no_think") else self.service.prompt_suffix
        self.arms = self.p_list("arms") or [a.name for a in self.config.arms_by_rank()]
        self.arms.sort(key=lambda a: self.config.arms[a].rank if a in self.config.arms else 0)
        for arm in self.arms:
            self.check_arm(arm, "arms")
        skills_path = self.p_str("skills")
        if not skills_path:
            raise ValueError("brick_skill needs skills=<skill table JSON> (eval/fit_skills.py)")
        table = load_skill_table(skills_path)
        self.latency = LatencyProfile.load(self.p_str("latency"), fallback=self.config.latency_defaults)
        for arm in self.arms:
            if arm not in table:
                raise ValueError(f"brick_skill: no skill vector for arm {arm!r} in {skills_path}")
            if arm not in self.latency.arms:
                raise ValueError(f"brick_skill: no latency profile for arm {arm!r} "
                                 "(pass latency=<profile JSON> or set latency_defaults in the config)")
        self.skills = {arm: table[arm] for arm in self.arms}
        base = brick_math.effective_params(self.p_float("r", 0.0))
        beta, lam = self.p_float("beta"), self.p_float("lam")
        self.brick = brick_math.BrickParams(mu=base.mu, b=base.b,
                                             beta=base.beta if beta is None else beta,
                                             lam=base.lam if lam is None else lam)
        self.tie_eps = self.p_float("tie_eps", brick_math.TIE_EPS)
        self.live_load = self.p_bool("live_load", True)
        self.capability_fn = self.res.capability_fn or get_classifier(
            self.p_str("capability", DEFAULT_MODEL_DIR), self.p_int("max_length", 512)).predict
        small, mid, large = _tiers(self.arms)
        self.fallback = {"easy": small or mid or large, "medium": mid or large, "hard": large}

    def latencies(self, prompt_chars: int, complexity: dict | None, max_tokens: int | None = None,
                  loads: dict | None = None) -> dict[str, dict[str, float]]:
        """Expected latency breakdown per candidate arm (loads: per backend model, or None)."""
        ptoks = self.latency.prompt_tokens(prompt_chars)
        out = {}
        for arm in self.arms:
            backend = self.config.arms[arm].model
            queue = 0.0
            if loads:
                sharing = [a for a in self.config.arms if self.config.arms[a].model == backend]
                queue = queue_wait_ms(loads.get(backend), self.latency.backend_service_ms(backend, sharing))
            out[arm] = expected_latency_ms(self.latency.arms[arm], ptoks, complexity, max_tokens, queue)
        return out

    def choose(self, complexity: dict | None, capability: dict, prompt_chars: int,
               max_tokens: int | None = None, loads: dict | None = None) -> tuple[str, str, dict]:
        """(arm, reason, info) — the pure scoring step, used offline by evaluate_router."""
        label, conf = brick_math.complexity_label(complexity)
        tau_q = brick_math.tau_query(label, conf)
        z_q = brick_math.required_level(tau_q, self.brick)
        p = brick_math.normalise_probs(capability)
        lats = self.latencies(prompt_chars, complexity, max_tokens, loads)
        arm, scores, margin = brick_math.select(p, self.skills, {a: v["total_ms"] for a, v in lats.items()},
                                                z_q, self.brick, self.tie_eps)
        top = max(p, key=p.get)
        best = next(s for s in scores if s.arm == arm)
        reason = (f"argmin J: {arm} J={best.J:.3f} (D={best.D:.3f}, lat {best.lat_ms:.0f} ms); "
                  f"{label} conf {conf:.2f} -> tau {tau_q:.3f}; top capability {top} {p[top]:.2f}")
        info = {"scores": {s.arm: {**s.to_dict(), "queue_ms": round(lats[s.arm]["queue_ms"], 1),
                                   "out_tokens": round(lats[s.arm]["out_tokens"], 1)} for s in scores},
                "margin": round(margin, 4), "tau_q": round(tau_q, 4), "z_q": round(z_q, 4),
                "complexity_label": label, "capability": {c: round(v, 4) for c, v in p.items()}}
        return arm, reason, info

    async def _complexity(self, text):
        t0 = time.perf_counter()
        try:
            probs = await signals.brick_probs(
                self.service.base_url, text, kind=self.service.kind, model=self.service.served_name,
                prompt_suffix=self.suffix, client=self.res.client)
            return probs, None, (time.perf_counter() - t0) * 1000
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
            return None, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000

    async def _capability(self, text):
        t0 = time.perf_counter()
        try:
            probs = await run_capability(self.capability_fn, text)
            return probs, None, (time.perf_counter() - t0) * 1000
        except Exception as e:  # noqa: BLE001 - any classifier failure falls back to complexity
            return None, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000

    async def decide(self, req):
        (cx, cx_err, brick_ms), (cap, cap_err, cap_ms) = await asyncio.gather(
            self._complexity(req.last_user_text()), self._capability(routing_text(req.messages)))
        sig = {"brick_ms": round(brick_ms, 2), "capability_ms": round(cap_ms, 2), "service": self.service.name,
               "params": {k: round(v, 6) for k, v in vars(self.brick).items()}, "complexity": cx}
        if cx_err:
            sig["complexity_error"] = cx_err
        if cap_err:
            label, _ = brick_math.complexity_label(cx)
            sig.update(capability_error=cap_err, error=cap_err, margin=0.0)
            return self.decision(self.fallback[label], f"capability error, complexity {label} fallback: {cap_err}",
                                 signals_=sig)
        loads = await self.res.load.get() if self.live_load else None
        if loads is not None:
            sig["load"] = loads
        arm, reason, info = self.choose(cx, cap, req.prompt_chars(), req.max_tokens, loads)
        scores = info.pop("scores")
        sig.update(info)
        return self.decision(arm, reason, scores=scores, signals_=sig)


# ─── wrappers ────────────────────────────────────────────────────────────

@register
class LoadAwareRouter(Router):
    """load_aware: nudge the inner decision by live backend load.

    Load is per backend model, so arms that share a backend (27B think on/off) share it.
    * chosen arm's backend full (in_flight >= capacity):
        - move to the nearest higher-rank arm whose backend is free, if margin <= margin_up
          (default: always);
        - else to the nearest lower-rank arm whose backend is free, if margin <= margin_down
          (default 0.1).
    * promote (default 0 = off): if the top arm's backend is idle and margin < promote,
      move a lower decision to the top arm.
    `margin` is the inner router's signals.margin (0 when absent). Unknown load = no change.
    brick_skill already prices the live queue into its latency term, so on top of it this
    wrapper only acts on full backends and borderline J gaps.
    """
    name = "load_aware"
    wrapper = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.margin_up = self.p_float("margin_up", 1.0)
        self.margin_down = self.p_float("margin_down", 0.1)
        self.promote = self.p_float("promote", 0.0)

    async def decide(self, req):
        d = await self.inner.decide(req)
        if d.model not in self.config.arms:
            return d
        loads = await self.res.load.get()
        d.signals["load"] = loads
        margin = float(d.signals.get("margin") or 0.0)
        order = [a.name for a in self.config.arms_by_rank()]
        i = order.index(d.model)

        def state(arm):
            return loads.get(self.config.arms[arm].model)

        def busy(arm):
            s = state(arm)
            return s is not None and s["in_flight"] >= s["capacity"]

        def free(arm):
            s = state(arm)
            return s is not None and s["in_flight"] < s["capacity"]

        up = next((a for a in order[i + 1:] if free(a)), None)
        down = next((a for a in reversed(order[:i]) if free(a)), None)
        new = None
        if busy(d.model):
            if up and margin <= self.margin_up:
                new, why = up, "busy, shifted up"
            elif down and margin <= self.margin_down:
                new, why = down, "busy, shifted down"
        elif self.promote > 0 and d.model != order[-1] and margin < self.promote:
            s = state(order[-1])
            if s is not None and s["in_flight"] == 0:
                new, why = order[-1], "top arm idle, promoted borderline decision"
        if new:
            d.signals["inner_model"] = d.model
            d.reason = f"load_aware: {d.model} {why} to {new} (margin {margin:.3f}; inner: {d.reason})"
            d.model = new
        return d


# ─── spec parsing / build ───────────────────────────────────────────────

def parse_spec(spec) -> list[tuple[str, dict]]:
    """Spec (string, JSON string, dict or list) -> [(name, params)] outer→inner."""
    if isinstance(spec, str) and spec.strip()[:1] in "[{":
        spec = json.loads(spec)
    if isinstance(spec, dict):
        layers = []
        while spec:
            layers.append((spec["router"], dict(spec.get("params", {}))))
            spec = spec.get("inner")
        return layers
    if isinstance(spec, list):
        return [(s["router"], dict(s.get("params", {}))) for s in spec]
    layers = []
    for part in str(spec).split(">"):
        name, _, rest = part.strip().partition(":")
        params = {}
        for kv in filter(None, (x.strip() for x in rest.split(","))):
            k, eq, v = kv.partition("=")
            if not eq:
                raise ValueError(f"bad router param {kv!r} (want key=value)")
            params[k.strip()] = v.strip()  # routers coerce via p_* helpers
        layers.append((name.strip(), params))
    return layers


def spec_label(layers: list[tuple[str, dict]]) -> str:
    def one(name, params):
        if not params:
            return name
        vals = ",".join(f"{k}={'|'.join(map(str, v)) if isinstance(v, list) else v}" for k, v in params.items())
        return f"{name}:{vals}"
    return ">".join(one(n, p) for n, p in layers)


def build(spec, config: Config, *, client: httpx.AsyncClient | None = None,
          load_ttl_s: float = 0.2, capability_fn=None) -> Router:
    """Build a router chain from a spec. ``capability_fn(text) -> {capability: prob}``
    replaces the local ModernBERT classifier (tests, or a remote classifier later)."""
    layers = parse_spec(spec)
    for i, (name, _) in enumerate(layers):
        if name not in ROUTERS:
            raise ValueError(f"unknown router {name!r} (have {sorted(ROUTERS)})")
        last = i == len(layers) - 1
        if ROUTERS[name].wrapper == last:
            raise ValueError(f"{name!r}: wrappers need an inner router; leaf routers must come last")
    res = Resources(config, client=client, load_ttl_s=load_ttl_s, capability_fn=capability_fn)
    router = None
    for name, params in reversed(layers):
        router = ROUTERS[name](config, params, res, inner=router)
    router.label = spec_label(layers)
    return router
