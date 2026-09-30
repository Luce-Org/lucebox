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
"""

from __future__ import annotations

import json
import time

import httpx

from . import signals
from .config import Config
from .types import RouteDecision, RouteRequest

ROUTERS: dict[str, type["Router"]] = {}


def register(cls):
    ROUTERS[cls.name] = cls
    return cls


class Resources:
    """Things routers share: one HTTP client and one cached load monitor."""

    def __init__(self, config: Config, client: httpx.AsyncClient | None = None, load_ttl_s: float = 0.2):
        self.config = config
        self.client = client or httpx.AsyncClient()
        self._own_client = client is None
        self.load = signals.LoadMonitor(config.models, self.client, ttl_s=load_ttl_s)

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

    def check_model(self, model: str | None, key: str):
        if model is not None and model not in self.config.models:
            raise ValueError(f"{self.name}: {key}={model!r} is not a configured model")

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
    """fixed:model=X — always X (baselines)."""
    name = "fixed"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.model = self.p_str("model")
        if not self.model:
            raise ValueError("fixed router needs model=")

    async def decide(self, req):
        return self.decision(self.model, "fixed")


def _default_tiers(config: Config) -> tuple[str | None, str | None, str]:
    names = [m.name for m in config.by_rank()]
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
    Params: service, t_small, t_mid, h_max, small, mid, large (small/mid may be
    "none"), no_think (llama-server kind only: append the empty think block),
    on_error (model used when the classifier fails; default large).
    signals.margin = distance of the probs from flipping the decision.
    """
    name = "brick"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        small, mid, large = _default_tiers(self.config)
        self.small = self.p_str("small", small)
        self.mid = self.p_str("mid", mid)
        self.large = self.p_str("large", large)
        self.t_small = self.p_float("t_small", 0.8)
        self.t_mid = self.p_float("t_mid", 0.5)
        self.h_max = self.p_float("h_max", 0.5)
        self.on_error = self.p_str("on_error", self.large)
        self.service = self.config.service(self.p_str("service", "brick-max"))
        self.suffix = signals.THINK_SUFFIX if self.p_bool("no_think") else self.service.prompt_suffix
        for key in ("small", "mid"):
            self.check_model(getattr(self, key), key)

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


# ─── wrappers ────────────────────────────────────────────────────────────

@register
class LoadAwareRouter(Router):
    """load_aware: nudge the inner decision by live backend load.

    * chosen model full (in_flight >= capacity):
        - shift one rank up if that model is free and margin <= margin_up (default: always);
        - else shift one rank down if free and margin <= margin_down (default 0.1).
    * promote (default 0 = off): if the largest model is idle and margin < promote,
      move a smaller decision to the largest model.
    `margin` is the inner router's signals.margin (0 when absent). Unknown load = no change.
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
        if d.model not in self.config.models:
            return d
        loads = await self.res.load.get()
        d.signals["load"] = loads
        margin = float(d.signals.get("margin") or 0.0)
        order = [m.name for m in self.config.by_rank()]
        i = order.index(d.model)

        def busy(m):
            s = loads.get(m)
            return s is not None and s["in_flight"] >= s["capacity"]

        def free(m):
            s = loads.get(m)
            return s is not None and s["in_flight"] < s["capacity"]

        up = order[i + 1] if i + 1 < len(order) else None
        down = order[i - 1] if i > 0 else None
        new = None
        if busy(d.model):
            if up and free(up) and margin <= self.margin_up:
                new, why = up, "busy, shifted up"
            elif down and free(down) and margin <= self.margin_down:
                new, why = down, "busy, shifted down"
        elif self.promote > 0 and d.model != order[-1] and margin < self.promote:
            s = loads.get(order[-1])
            if s is not None and s["in_flight"] == 0:
                new, why = order[-1], "largest idle, promoted borderline decision"
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
          load_ttl_s: float = 0.2) -> Router:
    """Build a router chain from a spec."""
    layers = parse_spec(spec)
    for i, (name, _) in enumerate(layers):
        if name not in ROUTERS:
            raise ValueError(f"unknown router {name!r} (have {sorted(ROUTERS)})")
        last = i == len(layers) - 1
        if ROUTERS[name].wrapper == last:
            raise ValueError(f"{name!r}: wrappers need an inner router; leaf routers must come last")
    res = Resources(config, client=client, load_ttl_s=load_ttl_s)
    router = None
    for name, params in reversed(layers):
        router = ROUTERS[name](config, params, res, inner=router)
    router.label = spec_label(layers)
    return router
