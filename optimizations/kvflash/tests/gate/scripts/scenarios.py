"""GPU regression scenarios for the compile-verified-only KVFlash session fixes.
usage (env KV_BIN, KV_ROOT, KV_PORT, KV_MODEL, optional KV_MMPROJ from run_scenarios.sh):  python3 -I scenarios.py 1|2|3|all

Every scenario runs a live server (LUCE_KVFLASH_RESTORE_CONSUME=1, LUCE_KVFLASH_SESSION default on, --kvflash 8192,
--max-ctx 131072, temp 0 seed 1), sends the setup requests then the probe request(s), and compares each probe
answer with a COLD reference: a fresh server, same flags, that sees only that probe (nothing to restore).
PASS = every probe byte-equal to its cold reference. The log lines proving the intended restore path engaged
([snap], pooled restore, live restore, resume) are printed as informational 'engaged' evidence.
 1  consume: session A past the pool (two turns), unrelated cold session B, then (P1) A extended by one turn,
    (P2) A's first request with an edit at ~9000 tokens.
 2  boundary 512: long prompt A (> pool), short prompt B (snapshot at the 512 boundary), probe P shares B's first
    ~640 tokens (>= 512) and diverges.
 3  image: text session A with a rung at 8192, unrelated image session I, probe T = text request sharing I's
    leading text (needs --mmproj; SKIP when KV_MMPROJ is unset/missing)."""
import base64, json, os, re, struct, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvlib
from kvlib import Server, chat, filler, ROOT

SYS = {"role": "system", "content": "You are a careful assistant who reads long record dumps. Keep answers short (two sentences)."}
MMPROJ = os.environ.get("KV_MMPROJ", "")
ENV = {"LUCE_KVFLASH_RESTORE_CONSUME": "1"}
ENGAGED = re.compile(r"\[snap\]|pooled restore|live restore|resume:|state ladder|boundary snapshot|chat CACHE|prefix_len=")


def png(w=256, h=256):
    raw = b"".join(b"\x00" + b"".join(bytes((x * 255 // w, y * 255 // h, (x + y) * 255 // (w + h))) for x in range(w)) for y in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def user(text):
    return {"role": "user", "content": text}


def run_live(name, extra, seq):
    """seq: list of (label, callable(history)->messages). Returns {label: (messages, answer)}."""
    out, hist = {}, {}
    log = f"{ROOT}/s{name}_live.log"
    with Server(log, env=ENV, extra=extra):
        chat([user(filler(1500, salt=99))], max_tokens=4)  # warm-up, same as the gates
        for label, mk in seq:
            msgs = mk(hist)
            r = chat(msgs, max_tokens=64)
            hist[label] = (msgs, r["text"])
            print(f"[s{name}] {label}: prompt_tokens={r['prompt_tokens']} ttft={r['ttft']:.2f}s answer={r['text'][:60]!r}", flush=True)
    return hist, [l.rstrip() for l in open(log, errors="replace") if ENGAGED.search(l)]


def cold(name, extra, label, msgs):
    with Server(f"{ROOT}/s{name}_cold_{label}.log", env=ENV, extra=extra):
        chat([user(filler(1500, salt=99))], max_tokens=4)
        return chat(msgs, max_tokens=64)["text"]


def verdict(name, extra, seq, probes):
    hist, eng = run_live(name, extra, seq)
    res = {}
    for p in probes:
        c = cold(name, extra, p, hist[p][0])
        res[p] = dict(live=hist[p][1], cold=c, equal=hist[p][1] == c)
        print(f"[s{name}] probe {p}: {'EQUAL' if res[p]['equal'] else 'DIFF'}", flush=True)
    json.dump(dict(results=res, engaged=eng), open(f"{ROOT}/s{name}.json", "w"), indent=1)
    print(f"[s{name}] engaged log lines:\n  " + "\n  ".join(eng[-40:]))
    ok = all(v["equal"] for v in res.values())
    print(f"SCENARIO {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def s1():
    a1 = lambda h: [SYS, user(filler(10000, salt=11) + "\nQuestion 1: what is the secret code? Answer in one sentence.")]
    a2 = lambda h: h["A1"][0] + [{"role": "assistant", "content": h["A1"][1]},
                                 user(" ".join(f"Update 2.{i}: record R2{i:03d} shows alpha-{i * 7 % 97}; status ok." for i in range(12)) + "\nQuestion 2: summarise update 2.0.")]
    b = lambda h: [SYS, user(filler(3000, salt=77) + "\nQuestion: what is the secret code?")]
    p1 = lambda h: h["A2"][0] + [{"role": "assistant", "content": h["A2"][1]}, user("Update 3.0: record R3000 shows alpha-5.\nQuestion 3: summarise update 3.0.")]
    def p2(h):
        m = json.loads(json.dumps(h["A1"][0]))
        assert "Record 606:" in m[1]["content"]  # ~14.9 tokens per record: edit lands near token 9000
        m[1]["content"] = m[1]["content"].replace("Record 606:", "Record 606 (EDITED):", 1)
        return m
    return verdict("1", [], [("A1", a1), ("A2", a2), ("B", b), ("P1", p1), ("P2", p2)], ["P1", "P2"])


def s2():
    btext = filler(1500, salt=22)
    cut = btext.index("Record 40:")  # ~640 tokens of shared prefix, then diverge
    a = lambda h: [SYS, user(filler(12000, salt=21) + "\nQuestion: what is the secret code?")]
    b = lambda h: [SYS, user(btext + "\nQuestion: what is the secret code?")]
    p = lambda h: [SYS, user(btext[:cut] + " DIVERGE. " + filler(900, salt=23) + "\nQuestion: what is the secret code?")]
    # two-turn variant: B2's history cut (first user message, ~650 tokens) is the snapshot point; P2 repeats that
    # first turn and diverges right after it, so the restore point is the 512 boundary below the cut.
    first = filler(600, salt=24) + "\nQuestion 1: what is the secret code?"
    b2 = lambda h: [SYS, user(first), {"role": "assistant", "content": "The secret code is QX-7319-ZEBRA."}, user(filler(800, salt=25) + "\nQuestion 2: repeat the code.")]
    p2 = lambda h: [SYS, user(first), {"role": "assistant", "content": "I could not find a code."}, user(filler(800, salt=26) + "\nQuestion 2: repeat the code.")]
    return verdict("2", [], [("A", a), ("B", b), ("P", p), ("B2", b2), ("P2", p2)], ["P", "P2"])


def s3():
    if not (MMPROJ and os.path.exists(MMPROJ)):
        print(f"SCENARIO 3: SKIP (KV_MMPROJ unset or missing: {MMPROJ!r})")
        return None
    extra = ["--mmproj", MMPROJ]
    img = "data:image/png;base64," + base64.b64encode(png()).decode()
    head = filler(700, salt=32)
    a = lambda h: [SYS, user(filler(10000, salt=31) + "\nQuestion: what is the secret code?")]
    i = lambda h: [SYS, {"role": "user", "content": [{"type": "text", "text": head}, {"type": "image_url", "image_url": {"url": img}},
                                                    {"type": "text", "text": "Describe the image in one sentence."}]}]
    t = lambda h: [SYS, user(head + "\nWhat is 2+2? Answer with one number.")]
    return verdict("3", extra, [("A", a), ("I", i), ("T", t)], ["T"])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"1": s1, "2": s2, "3": s3}
    sel = fns if which == "all" else {which: fns[which]}
    rc = [f() for f in sel.values()]
    sys.exit(0 if all(r is not False for r in rc) else 1)
