#!/usr/bin/env python3
"""Compact viewer for terminus trajectory.jsonl: shows selected agent steps with think len, analysis, commands."""
import json, sys, re
def steps(path):
    L=[json.loads(l) for l in open(path)]
    return [x for x in L if x.get('source')=='agent']
def show(path, which='auto', n_head=3, n_tail=3, maxcmd=5, obs=0):
    ag=steps(path)
    print(f"=== {path}  agent_steps={len(ag)}")
    if which=='all': sel=ag
    elif which=='auto':
        mid=len(ag)//2
        sel=ag[:n_head]+ag[mid:mid+1]+ag[-n_tail:]
    else: sel=[ag[i] for i in which if i<len(ag)]
    seen=set()
    for x in sel:
        if x['step_id'] in seen: continue
        seen.add(x['step_id'])
        msg=x.get('message','') or ''
        ana=re.search(r'Analysis:\s*(.*?)(?:\nPlan:|$)',msg,re.S); plan=re.search(r'Plan:\s*(.*)',msg,re.S)
        ana=(ana.group(1) if ana else msg)[:300].replace('\n',' ')
        plan=(plan.group(1) if plan else '')[:300].replace('\n',' ')
        tcs=x.get('tool_calls') or []
        done=any(tc.get('function_name')=='task_complete' or (tc.get('arguments') or {}).get('task_complete') for tc in tcs)
        print(f"--step {x['step_id']} think={len(x.get('reasoning_content') or '')} cmds={len(tcs)}")
        print("  A:",ana); 
        if plan: print("  P:",plan)
        for tc in tcs[:maxcmd]:
            a=tc.get('arguments') or {}
            ks=a.get('keystrokes') or json.dumps(a)[:120]
            print("   $",ks[:160].replace('\n','⏎'))
        if obs:
            o=x.get('observation') or {}
            for r in (o.get('results') or [])[:1]:
                print("  OBS:",(r.get('content') or '')[-obs:].replace('\n',' | '))
if __name__=='__main__':
    p=sys.argv[1]; w=sys.argv[2] if len(sys.argv)>2 else 'auto'
    if w not in('auto','all'): w=[int(i) for i in w.split(',')]
    obs=int(sys.argv[3]) if len(sys.argv)>3 else 0
    show(p,w,obs=obs)
