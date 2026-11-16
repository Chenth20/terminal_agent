import asyncio, aiohttp, json, time, sys, random
URL="https://envarena-prod.alipay.com/api/v1/chat/completions"; H={"Authorization":"Bearer areal","Content-Type":"application/json"}
CONC=int(sys.argv[1]); DUR=int(sys.argv[2]); MAXTOK=int(sys.argv[3]); CTX_REPEAT=int(sys.argv[4]) if len(sys.argv)>4 else 60
CTX="You are a terminal agent working in a repo. Transcript so far:\n" + ("$ ls -la /workspace\ntotal 48\ndrwxr-xr-x 6 root root 4096 Jan 1 00:00 .\n-rw-r--r-- 1 root root 1024 Jan 1 00:00 Makefile\n$ cat Makefile\nall: build\n\tgcc -O2 -o app main.c util.c\n"*CTX_REPEAT)
TASKS=["Implement a bash script that finds duplicate files by sha256 under /workspace and prints groups.","Write a C function that parses an INI file into a hash map; explain edge cases.","Plan step by step how to debug a segfault in app when input is empty; then give the gdb commands.","Write a Python script that tails a log and alerts on 5 errors/minute, with tests."]
stats=dict(ok=0,fail=0,toks=0,ttft=[],dur=[],errs={})
async def worker(s, wid, deadline):
    while time.time()<deadline:
        t0=time.time(); ttft=None; toks=0
        body={"model":"qwen35-4b-chen-ais","messages":[{"role":"user","content":CTX+f"\n(session {wid}-{random.randint(0,10**6)}) {random.choice(TASKS)}"}],"max_tokens":MAXTOK,"stream":True,"stream_options":{"include_usage":True},"reasoning_effort":"xhigh"}
        try:
            async with s.post(URL,headers=H,json=body,timeout=aiohttp.ClientTimeout(total=1800)) as r:
                if r.status!=200:
                    e=f"HTTP {r.status}"; stats["errs"][e]=stats["errs"].get(e,0)+1; stats["fail"]+=1; await asyncio.sleep(2); continue
                async for line in r.content:
                    line=line.decode().strip()
                    if not line.startswith("data:") or line.endswith("[DONE]"): continue
                    d=json.loads(line[5:])
                    if ttft is None: ttft=time.time()-t0
                    if d.get("usage"): toks=d["usage"].get("completion_tokens",0)
            stats["ok"]+=1; stats["toks"]+=toks; stats["ttft"].append(ttft or 0); stats["dur"].append(time.time()-t0)
        except Exception as e:
            k=type(e).__name__; stats["errs"][k]=stats["errs"].get(k,0)+1; stats["fail"]+=1; await asyncio.sleep(2)
async def main():
    t0=time.time(); deadline=t0+DUR
    conn=aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as s:
        await asyncio.gather(*[worker(s,i,deadline) for i in range(CONC)])
    wall=time.time()-t0
    tt=sorted(stats["ttft"]); du=sorted(stats["dur"])
    p=lambda a,q: a[min(len(a)-1,int(len(a)*q))] if a else float('nan')
    print(f"CONC={CONC} dur={DUR}s maxtok={MAXTOK} ctx~{len(CTX)//3}tok | ok={stats['ok']} fail={stats['fail']} errs={stats['errs']} | completion tok/s={stats['toks']/wall:.0f} | ttft p50={p(tt,.5):.1f} p95={p(tt,.95):.1f}s | req dur p50={p(du,.5):.1f} p95={p(du,.95):.1f}s")
asyncio.run(main())
