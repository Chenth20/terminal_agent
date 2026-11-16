import asyncio, aiohttp, json, time, sys
URL="https://envarena-prod.alipay.com/api/v1/chat/completions"; H={"Authorization":"Bearer areal","Content-Type":"application/json"}
N=int(sys.argv[1]); MAXTOK=int(sys.argv[2])
PROMPTS=[f"Write a bash one-liner that counts lines in all .py files under a directory (variant {i}). Think briefly." for i in range(N)]
async def one(s,i):
    t0=time.time(); toks=0; ttft=None; err=None
    body={"model":"qwen35-4b-chen-ais","messages":[{"role":"user","content":PROMPTS[i]}],"max_tokens":MAXTOK,"stream":True,"stream_options":{"include_usage":True},"reasoning_effort":"xhigh"}
    try:
        async with s.post(URL,headers=H,json=body,timeout=aiohttp.ClientTimeout(total=900)) as r:
            if r.status!=200: return dict(i=i,err=f"HTTP {r.status} {(await r.text())[:120]}")
            async for line in r.content:
                line=line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d=json.loads(line[5:]); 
                if ttft is None: ttft=time.time()-t0
                if d.get("usage"): toks=d["usage"].get("completion_tokens",0)
    except Exception as e: err=repr(e)[:120]
    return dict(i=i,ttft=ttft,toks=toks,dur=time.time()-t0,err=err)
async def main():
    t0=time.time()
    async with aiohttp.ClientSession() as s: res=await asyncio.gather(*[one(s,i) for i in range(N)])
    wall=time.time()-t0; ok=[r for r in res if not r.get("err")]; tot=sum(r["toks"] for r in ok)
    print(f"N={N} ok={len(ok)} fail={len(res)-len(ok)} wall={wall:.1f}s total_completion_tokens={tot} agg_throughput={tot/wall:.1f} tok/s")
    if ok:
        tt=sorted(r["ttft"] for r in ok if r["ttft"]); du=sorted(r["dur"] for r in ok)
        print(f"ttft p50={tt[len(tt)//2]:.1f}s p95={tt[int(len(tt)*.95)-1]:.1f}s | dur p50={du[len(du)//2]:.1f}s max={du[-1]:.1f}s | per-req tok/s p50={sorted(r['toks']/r['dur'] for r in ok)[len(ok)//2]:.1f}")
    for r in res:
        if r.get("err"): print("  ERR", r["i"], r["err"])
asyncio.run(main())
