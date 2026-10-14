"""Shared helpers for KVFlash GPU gates. Python 3 stdlib only. Run with python3 -I."""
import json, os, subprocess, time, urllib.request, signal, re

ROOT = os.environ["KV_ROOT"]  # per-run output dir (required)
BIN = os.environ["KV_BIN"]  # luce_server binary (required)
MODEL = os.environ["KV_MODEL"]  # target model .gguf (required)
MAX_CTX = os.environ.get("KV_MAX_CTX", "131072")
POOL = os.environ.get("KV_POOL", "8192")
PORT = int(os.environ.get("KV_PORT", "18090"))
URL = f"http://127.0.0.1:{PORT}"


class Server:
    def __init__(self, log, env=None, extra=()):
        self.log, self.env, self.extra, self.p = log, env or {}, list(extra), None

    def __enter__(self):
        e = dict(os.environ)
        e["TMPDIR"] = os.environ["TMPDIR"]
        for k in list(e):
            if k.startswith("LUCE_KVFLASH_"):
                del e[k]
        e.update(self.env)
        args = [BIN, MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--max-ctx", MAX_CTX,
                "--kvflash", POOL, "--model-name", "luce"] + os.environ.get("KV_EXTRA", "").split() + self.extra
        self.f = open(self.log, "w")
        self.p = subprocess.Popen(args, stdout=self.f, stderr=subprocess.STDOUT, env=e,
                                  stdin=subprocess.DEVNULL)
        for _ in range(300):
            if self.p.poll() is not None:
                raise RuntimeError("server died; see " + self.log)
            try:
                urllib.request.urlopen(URL + "/v1/models", timeout=2)
                return self
            except Exception:
                time.sleep(2)
        raise RuntimeError("server start timeout")

    def __exit__(self, *a):
        if self.p and self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(30)
            except Exception:
                self.p.kill()
                self.p.wait()
        self.f.close()


def chat(messages, max_tokens=64, seed=1):
    """Streaming chat, temperature 0. Returns dict(text, ttft, total, completion_tokens, prompt_tokens, tps)."""
    body = json.dumps({"model": "luce", "messages": messages, "temperature": 0, "seed": seed,
                       "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text = []
    usage = {}
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices", []):
                delta = c.get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(delta)
                    n_chunks += 1
    total = time.time() - t0
    ct = usage.get("completion_tokens", n_chunks)
    dec = total - (ttft or 0)
    return dict(text="".join(text), ttft=ttft, total=total, completion_tokens=ct,
                prompt_tokens=usage.get("prompt_tokens"),
                tps=(ct - 1) / dec if dec > 0 and ct > 1 else None)


def filler(n_tokens, salt=0, needle=True):
    """Deterministic filler of ~n_tokens tokens; salt makes prompts distinct from token 0."""
    r = n_tokens // 16
    out = [f"Salt {salt}. "]
    for i in range(r):
        if needle and i == r // 2:
            out.append("NEEDLE: the secret code is QX-7319-ZEBRA. ")
        out.append(f"Record {i}: alpha-{(i * 7 + salt) % 97} bravo-{(i * 13 + salt * 3) % 89}. ")
        if i % 8 == 7:
            out.append("\n")
    return "".join(out)


def grep_log(path, pat):
    rx = re.compile(pat)
    return [l.rstrip("\n") for l in open(path, errors="replace") if rx.search(l)]
