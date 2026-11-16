#!/usr/bin/env python3
"""Per-engine concurrency sweep: find the largest concurrency that runs without KV retracts / queueing.

Workload mimics the TB2.1 agent mix: every request carries a long prompt (CTX_TOK tokens, mostly a shared prefix
so the radix cache helps as it does in real runs) and generates up to MAXTOK tokens. For each concurrency level
we keep exactly C requests in flight for DUR seconds, then report:
  completion tok/s, request dur p50/p95, ttft p50/p95, prompt_tokens as reported by the server, and the number of
  "KV cache pool is full. Retract requests" / "#queue-req: N>0" events in the engine's sglang log during the window.

  python3 stress_kv.py http://IP:PORT served-name /path/engine.sglang.log CTX_TOK MAXTOK DUR C1 C2 C3 ...
"""
import asyncio, aiohttp, json, time, sys, random, os, re, statistics as st

URL, MODEL, ELOG = sys.argv[1].rstrip("/") + "/v1/chat/completions", sys.argv[2], sys.argv[3]
CTX_TOK, MAXTOK, DUR = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
LEVELS = [int(x) for x in sys.argv[7:]]
H = {"Content-Type": "application/json"}
CHUNK = ("$ ls -la /workspace\ntotal 48\ndrwxr-xr-x 6 root root 4096 Jan 1 00:00 .\n-rw-r--r-- 1 root root 1024 Jan 1 00:00 Makefile\n"
         "$ cat Makefile\nall: build\n\tgcc -O2 -o app main.c util.c\n$ ./app < /dev/null\nSegmentation fault (core dumped)\n$ gdb -batch -ex run -ex bt ./app\n#0 0x0000 in parse (p=0x0) at util.c:42\n")
# measured on this tokenizer: CHUNK (280 chars) = ~120 tokens. Each worker is one "agent session": its own
# prefix (unique salt in every chunk, so no radix sharing ACROSS workers, like distinct tasks) reused across its steps.
NCHUNK = int(CTX_TOK * 0.9 / 120) + 1
def prefix_for(wid):
    salt = f"{wid:04d}-{random.randint(0, 10**9):09d}"
    return "".join(CHUNK.replace("Jan 1 00:00", f"sess {salt} {i:05d}") for i in range(NCHUNK))
PREFIXES = {}
TAILS = ["Implement a bash script that finds duplicate files by sha256 under /workspace and prints groups.",
         "Write a C function that parses an INI file into a hash map; explain edge cases.",
         "Plan step by step how to debug the segfault above; then give the gdb commands.",
         "Write a Python script that tails a log and alerts on 5 errors/minute, with tests."]
SYS = "You are a terminal agent. Respond with JSON {analysis, plan, commands:[{keystrokes,duration}], task_complete}."

def log_counts(path, t0):
    """retracts and queue>0 decode lines in ELOG with timestamp >= t0 (local time)."""
    ret = q = 0; dec = 0; kvmax = 0.0; runmax = 0
    tstart = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
    try:
        with open(path, errors="ignore") as f:
            for l in f:
                if l[:19] < tstart: continue
                if "Retract requests" in l: ret += 1
                if "Decode batch" in l:
                    dec += 1
                    m = re.search(r"#queue-req: (\d+)", l); 
                    if m and int(m.group(1)) > 0: q += 1
                    m = re.search(r"token usage: ([\d.]+)", l)
                    if m: kvmax = max(kvmax, float(m.group(1)))
                    m = re.search(r"#running-req: (\d+)", l)
                    if m: runmax = max(runmax, int(m.group(1)))
    except FileNotFoundError: pass
    return ret, q, dec, kvmax, runmax

async def one(s, wid, stats):
    t0 = time.time(); ttft = None; ct = pt = 0
    tail = CHUNK * random.randint(1, 3) + f"\n(session {wid}-{random.randint(0, 10**6)}) {random.choice(TAILS)}"
    if wid not in PREFIXES: PREFIXES[wid] = prefix_for(wid)
    body = {"model": MODEL, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": PREFIXES[wid] + tail}],
            "max_tokens": MAXTOK, "stream": True, "stream_options": {"include_usage": True},
            "temperature": 1, "top_p": 0.95, "top_k": 20}
    try:
        async with s.post(URL, headers=H, json=body, timeout=aiohttp.ClientTimeout(total=3600)) as r:
            if r.status != 200:
                stats["errs"][f"HTTP {r.status}"] = stats["errs"].get(f"HTTP {r.status}", 0) + 1; stats["fail"] += 1; return
            async for line in r.content:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                if ttft is None and d.get("choices") and (d["choices"][0].get("delta") or {}).get("content") is not None: ttft = time.time() - t0
                if d.get("usage"): ct = d["usage"].get("completion_tokens", 0); pt = d["usage"].get("prompt_tokens", 0)
        stats["ok"] += 1; stats["toks"] += ct; stats["ttft"].append(ttft or 0); stats["dur"].append(time.time() - t0); stats["pt"].append(pt); stats["ct"].append(ct)
    except Exception as e:
        k = type(e).__name__; stats["errs"][k] = stats["errs"].get(k, 0) + 1; stats["fail"] += 1

async def level(C):
    PREFIXES.clear()
    stats = dict(ok=0, fail=0, toks=0, ttft=[], dur=[], pt=[], ct=[], errs={})
    t0 = time.time(); deadline = t0 + DUR
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        async def worker(wid):
            while time.time() < deadline: await one(s, wid, stats)
        await asyncio.gather(*[worker(i) for i in range(C)])
    wall = time.time() - t0
    ret, q, dec, kvmax, runmax = log_counts(ELOG, t0)
    p = lambda a, qq: a[min(len(a) - 1, int(len(a) * qq))] if a else float("nan")
    tt = sorted(stats["ttft"]); du = sorted(stats["dur"])
    verdict = "OK" if ret == 0 and q == 0 else ("RETRACT" if ret else "QUEUE")
    print(f"C={C:3d} | ok={stats['ok']:4d} fail={stats['fail']} {stats['errs'] or ''} | prompt_tok p50={p(sorted(stats['pt']),.5):.0f} compl p50={p(sorted(stats['ct']),.5):.0f} "
          f"| compl tok/s={stats['toks']/wall:.0f} | ttft p50={p(tt,.5):.1f}s p95={p(tt,.95):.1f}s | dur p50={p(du,.5):.0f}s p95={p(du,.95):.0f}s "
          f"| engine: retracts={ret} queue>0 lines={q}/{dec} kv_max={kvmax:.2f} running_max={runmax} -> {verdict}", flush=True)
    return verdict

async def main():
    print(f"url={URL} model={MODEL} ctx~{CTX_TOK} maxtok={MAXTOK} dur={DUR}s levels={LEVELS}", flush=True)
    for C in LEVELS:
        v = await level(C)
        await asyncio.sleep(20)  # let the engine drain between levels
asyncio.run(main())
