/**
 * Client prefix diagnostics and Lucebox per-chat cache policy.
 * Client character-prefix stability is diagnostic only. The server's rendered
 * effective tokens and verified backend snapshot position determine actual reuse.
 * /cache exact|auto|status|release controls this chat; exact is the rollout default.
 * Explicit <lucebox_bulk>...</lucebox_bulk> document regions may compress in auto.
 */

import { createHash, randomUUID } from "node:crypto";
import * as os from "node:os";
import * as path from "node:path";
import * as fs from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { convertToLlm } from "@earendil-works/pi-coding-agent";

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------
const LOG_ENABLED = process.env.PCG_LOG !== "0";
const LOG_TEXT = process.env.PCG_LOG_TEXT === "1";
const NOTIFY_ON_BREAK = process.env.PCG_NOTIFY === "1";
const DEFER_COMPACTION = process.env.PCG_DEFER === "1";
const LOG_FILE =
  process.env.PCG_LOG_FILE || path.join(os.tmpdir(), "prefix-cache-guard.log");

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/** Serialize the outgoing messages the same way pi sends them, into a stable
 *  byte string we can compare across calls.
 *
 *  IMPORTANT: we join per-message JSON with a NUL separator rather than
 *  JSON.stringify-ing the whole array. If we stringified the array, the
 *  previous call's closing `]` would sit where the current call has `},{`,
 *  so `startsWith` would be false even for a perfectly append-only history.
 *  Per-message + separator makes a stable history a true byte-prefix.
 *  (NUL is safe: JSON.stringify escapes any literal NUL in content to the
 *  6-char `\u0000`, so it never collides with the separator.) */
function canonical(messages: unknown[]): string {
  const llm = convertToLlm(messages as any);
  return llm.map((m) => JSON.stringify(m)).join("\u0000");
}

function sha(s: string): string {
  return createHash("sha256").update(s).digest("hex").slice(0, 16);
}

/** Length of the shared prefix of two strings (char count). */
function longestCommonPrefix(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a.charCodeAt(i) === b.charCodeAt(i)) i++;
  return i;
}

function appendLog(line: string): void {
  try {
    fs.appendFileSync(LOG_FILE, line + "\n");
  } catch {
    /* never let logging break the agent loop */
  }
}

interface Reading {
  ts: number;
  msgs: number;
  chars: number;
  hash: string;
  status: "baseline" | "stable" | "break" | "reset";
  detail: string;
}

// ---------------------------------------------------------------------------
// extension
// ---------------------------------------------------------------------------
export default function (pi: ExtensionAPI) {
  let prev: { text: string; hash: string; n: number } | null = null;
  const recent: Reading[] = [];

  function remember(r: Reading): void {
    recent.push(r);
    if (recent.length > 50) recent.shift();
  }

  // --- 1. DIAGNOSTIC: is the outgoing prefix a growing prefix of last call? ---
  pi.on("context", (event, ctx) => {
    const text = canonical(event.messages);
    const hash = sha(text);
    const n = event.messages.length;

    if (!prev) {
      if (LOG_ENABLED)
        appendLog(
          `[pcg] baseline  msgs=${n} chars=${text.length} sha=${hash}`,
        );
      remember({ ts: Date.now(), msgs: n, chars: text.length, hash, status: "baseline", detail: "first call" });
      prev = { text, hash, n };
      return;
    }

    const appendOnly = text.startsWith(prev.text);

    if (appendOnly) {
      const grew = text.length - prev.text.length;
      if (LOG_ENABLED)
        appendLog(
          `[pcg] STABLE    +${grew} chars  (now ${text.length}, msgs=${n})  sha=${hash}  -> inspect server token reuse`,
        );
      remember({ ts: Date.now(), msgs: n, chars: text.length, hash, status: "stable", detail: `+${grew} chars` });
    } else {
      const lcp = longestCommonPrefix(prev.text, text);
      const pct = prev.text.length ? (100 * lcp) / prev.text.length : 0;
      const at = lcp;
      const ctxw = 80;
      const prevSlice = prev.text.slice(Math.max(0, at - ctxw), at + ctxw);
      const curSlice = text.slice(Math.max(0, at - ctxw), at + ctxw);
      const line =
        `[pcg] BREAK stable prefix=${lcp} chars of ${prev.text.length} prev (${pct.toFixed(1)}%) msgs ${prev.n}->${n} sha=${hash}` +
        (LOG_TEXT ? `\n          prev@${at}: ...${JSON.stringify(prevSlice)}\n          cur @${at}: ...${JSON.stringify(curSlice)}` : "");
      if (LOG_ENABLED) appendLog(line);
      if (NOTIFY_ON_BREAK)
        ctx.ui.notify(`[pcg] prefix broke at char ${lcp} (${pct.toFixed(1)}% of prev)`, "warning");
      remember({ ts: Date.now(), msgs: n, chars: text.length, hash, status: "break", detail: `stable ${lcp}/${prev.text.length} (${pct.toFixed(1)}%)` });
    }

    prev = { text, hash, n };
  });

  // --- 2. COMPACTION GUARD: log the one-time prefix reset; optionally defer ---
  pi.on("session_before_compact", (event, ctx) => {
    const { preparation, reason } = event;
    const tokens = preparation.tokensBefore?.toLocaleString() ?? "?";
    if (LOG_ENABLED)
      appendLog(
        `[pcg] COMPACTION  reason=${reason} tokensBefore=${tokens} firstKept=${preparation.firstKeptEntryId ?? "?"}`,
      );
    // A compaction rebuilds the context as [summary + kept], so the next call
    // will NOT be a prefix of the last. Mark the baseline so we don't flag the
    // immediate post-compaction call as a spurious "break".
    prev = null;
    remember({ ts: Date.now(), msgs: 0, chars: 0, hash: "-", status: "reset", detail: `compaction (${reason})` });

    if (DEFER_COMPACTION && reason === "threshold") {
      // Only defer *proactive* compaction. If we actually overflow, pi retries
      // with reason="overflow", which we never cancel. Safe to defer here.
      ctx.ui.notify("[pcg] deferring proactive compaction to preserve prefix", "info");
      return { cancel: true };
    }
    return;
  });

  // --- on-demand status ---
  pi.registerCommand("pcg", {
    description: "Show recent prefix-cache stability readings",
    handler: async (_args, ctx) => {
      if (recent.length === 0) {
        ctx.ui.notify("[pcg] no readings yet", "info");
        return;
      }
      const lines = recent.slice(-10).map((r) => {
        const t = new Date(r.ts).toLocaleTimeString();
        return `${t}  ${r.status.padEnd(8)} ${r.detail}`;
      });
      ctx.ui.notify("[pcg] recent:\n" + lines.join("\n"), "info");
    },
  });
  // Reloads and visits to existing branches retain their owner and policy.
  // Rewinding to an ancestor (or adding a branch summary) creates a new branch.
  let cacheMode: "exact" | "auto" = "exact";
  let branchId = "main";
  let temporaryPurpose = false;
  const stateType = "lucebox-cache-policy-v1";
  const owner = (ctx: any) => `${ctx.sessionManager.getSessionId()}:${branchId}`;
  const savePolicy = () => pi.appendEntry(stateType, { mode: cacheMode, branch: branchId });
  const readPolicy = (entries: any[]) => {
    let policy: { mode: "exact" | "auto"; branch: string } | undefined;
    for (const entry of entries) {
      const data = entry.data;
      if (entry.type === "custom" && entry.customType === stateType &&
          (data?.mode === "exact" || data?.mode === "auto") &&
          typeof data.branch === "string" && data.branch.length > 0) {
        policy = { mode: data.mode, branch: data.branch };
      }
    }
    return policy;
  };
  pi.on("session_start", (_event, ctx) => {
    const policy = readPolicy(ctx.sessionManager.getBranch());
    cacheMode = policy?.mode ?? "exact"; branchId = policy?.branch ?? "main";
    temporaryPurpose = false; prev = null;
    // Anchor the first branch so a later visit can recover its policy.
    if (!policy) savePolicy();
  });
  pi.on("session_tree", (event, ctx) => {
    const previousBranch = branchId;
    const policy = readPolicy(ctx.sessionManager.getBranch());
    const sameLeaf = event.newLeafId !== undefined && event.newLeafId === event.oldLeafId;
    const oldPath = event.oldLeafId ? ctx.sessionManager.getBranch(event.oldLeafId) : [];
    const rewind = event.newLeafId === null || oldPath.some(entry => entry.id === event.newLeafId);
    const existingBranch = !event.summaryEntry && policy &&
      (sameLeaf || (!rewind && policy.branch !== previousBranch));
    cacheMode = policy?.mode ?? "exact";
    branchId = existingBranch ? policy.branch : randomUUID();
    temporaryPurpose = false; prev = null;
    if (!existingBranch) savePolicy();
  });
  pi.on("session_before_tree", () => { temporaryPurpose = true; });
  pi.on("session_before_compact", (event) => {
    temporaryPurpose = !(DEFER_COMPACTION && event.reason === "threshold");
  });
  pi.on("session_compact", () => { temporaryPurpose = false; });
  pi.on("session_compact_failed", () => { temporaryPurpose = false; });
  pi.on("agent_start", () => { temporaryPurpose = false; });

  // Compaction bypasses the payload hook in this Pi version, but passes through
  // provider header transformation. The Lucebox gateway normalizes these headers.
  pi.on("before_provider_headers", (event, ctx) => {
    if (ctx.model?.provider !== "lucebox") return;
    event.headers["x-lucebox-cache-owner"] = owner(ctx);
    event.headers["x-lucebox-cache-purpose"] = temporaryPurpose ? "compaction" : "chat";
    event.headers["x-lucebox-cache-mode"] = cacheMode;
  });
  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== "lucebox") return;
    const payload = event.payload as any;
    if (!payload || !Array.isArray(payload.messages)) return;
    const summaries: number[] = [];
    const bulk: Array<{ message: number; part?: number; start: number; end: number; text: string }> = [];
    payload.messages.forEach((message: any, index: number) => {
      const parts = typeof message.content === "string" ? [{ part: undefined, text: message.content }] :
        Array.isArray(message.content) ? message.content.flatMap((block: any, part: number) =>
          block?.type === "text" && typeof block.text === "string" ? [{ part, text: block.text }] : []) : [];
      if (parts.some(({ text }: { text: string }) =>
          text.startsWith("The conversation history before this point was compacted into the following summary:\n\n<summary>\n") ||
          text.startsWith("The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"))) {
        summaries.push(index); return;
      }
      if (message.role !== "user") return;
      for (const { part, text } of parts) {
        const pattern = /<lucebox_bulk>([\s\S]*?)<\/lucebox_bulk>/g;
        for (const match of text.matchAll(pattern)) {
          if (!match[1]) continue;
          const start = match.index! + "<lucebox_bulk>".length;
          bulk.push({ message: index, ...(part === undefined ? {} : { part }),
            start: Buffer.byteLength(text.slice(0, start), "utf8"),
            end: Buffer.byteLength(text.slice(0, start + match[1].length), "utf8"), text: match[1] });
        }
      }
    });
    return { ...payload, extra_body: { ...payload.extra_body, lucebox_cache: {
      ...payload.extra_body?.lucebox_cache, owner: owner(ctx), mode: cacheMode,
      purpose: temporaryPurpose ? "compaction" : "chat",
      generation: ctx.sessionManager.getLeafId(), summary_messages: summaries, bulk_regions: bulk,
    } } };
  });
  pi.registerCommand("cache", {
    description: "Lucebox cache: exact, auto, status, or release",
    handler: async (args, ctx) => {
      const action = args.trim() || "status";
      if (action === "exact" || action === "auto") {
        cacheMode = action; savePolicy();
        ctx.ui.notify(`Lucebox cache policy: ${action}`, "info"); return;
      }
      if (action !== "status" && action !== "release") {
        ctx.ui.notify("Usage: /cache exact|auto|status|release", "warning"); return;
      }
      if (ctx.model?.provider !== "lucebox") {
        ctx.ui.notify("Select a Lucebox model first.", "warning"); return;
      }
      try {
        const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
        if (!auth.ok) throw new Error(auth.error);
        const url = new URL(auth.baseUrl || ctx.model.baseUrl);
        url.pathname = url.pathname.replace(/\/v1\/?$/, "") + `/cache/${action}`;
        url.search = "";
        const headers: Record<string,string> = { ...(auth.headers as any), "Content-Type": "application/json" };
        if (auth.apiKey) headers.Authorization = `Bearer ${auth.apiKey}`;
        const response = await fetch(url, { method: action === "release" ? "POST" : "GET", headers,
          body: action === "release" ? JSON.stringify({ owner: owner(ctx), model: ctx.model.id }) : undefined,
          signal: AbortSignal.timeout(15000) });
        if (!response.ok) throw new Error(`Cache operation failed (${response.status})`);
        const result = await response.json() as any;
        ctx.ui.notify(action === "release" ? "Released this chat's protected checkpoint." :
          `Policy: ${cacheMode}\nProtected chats: ${result.owners?.length ?? 0}\nLast path: ${result.last_request?.selected_path ?? "none"}`, "info");
      } catch (error) { ctx.ui.notify(String(error), "warning"); }
    },
  });
  // Deliberately no release on session_shutdown: it also fires during switches/reloads.

}
