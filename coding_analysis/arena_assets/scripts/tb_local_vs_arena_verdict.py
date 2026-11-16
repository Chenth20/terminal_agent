import csv,json,glob,re,os,collections
A='/storage/openpsi/users/chen/output'; JOB='cth-tb21m-mask-const-c312-2611051616'
A_s={}
for r in csv.DictReader(open(f'{A}/{JOB}/summary.tsv'),delimiter='\t'):
    A_s[(r['env_key'][4:],int(r['sample']))]=(int(r['score'] or 0),r['status'],int(r['agent_ms'] or 0)//1000)
def loc(run):
    L={}
    for p in glob.glob(f'runs/{run}/*/s*/result.json'):
        d=json.load(open(p)); L[(d['task'],d['sample'])]=(p,d)
    return L
def loc_reason(p,d):
    try: ex=str(json.load(open(p.replace('result.json','out/reward_result.json'))).get('raw',{}).get('log_excerpt',''))
    except Exception: ex=''
    f=re.search(r'FAILED [^ ]+::(\w+)',ex)
    return f.group(1) if f else None, ex
# arena per-function source lines (only from FAILURES blocks)
def arena_src(task, fn):
    lines=set()
    for f in glob.glob(f'{A}/cth-tb21*/tasks/tb2-{task}/s*/tbench_test.log'):
        t=open(f,errors='ignore').read()
        if '= FAILURES =' not in t or fn not in t: continue
        blk=t.split('= FAILURES =')[1]
        blk=re.split(r'\n=+ (?:warnings summary|PASSES|short test summary)',blk)[0]
        cur=None
        for l in blk.splitlines():
            m=re.match(r'^_+ (\w+)(?:\[.*\])? _+$',l)
            if m: cur=m.group(1); continue
            if cur!=fn: continue
            if l.startswith('_ _ _'): cur=None; continue
            if l.startswith('    ') or l.startswith('>   '):
                s=l[4:].rstrip().strip()
                if s and not s.startswith(('E ','assert','where','+','-','^','|','>>>','..')) and len(s)>10: lines.add(s)
    return lines
def local_src(task):
    s=set()
    for f in glob.glob(f'tasks/{task}/tests/*.py'):
        for l in open(f,errors='ignore'): 
            l=l.strip()
            if l: s.add(l)
    return s
for run in ['mask-c312-262k-local','mask-c312-96k-local']:
    L=loc(run); print(f'\n########## {run}')
    for k in sorted(set(A_s)&set(L)):
        p,d=L[k]; ascore=A_s[k][0]
        if ascore==d['score']: continue
        task,s=k; fn,ex=loc_reason(p,d)
        verdict='?'
        if ascore>d['score']:
            if d['harness_status']!='OK': verdict='INFRA('+d['harness_status']+')'
            elif d['agent_sec']>=14300: verdict='TTL'
            elif fn:
                asrc=arena_src(task,fn); lsrc=local_src(task)
                new=[x for x in asrc if x not in lsrc]
                verdict=f'TESTDIFF({len(new)} lines)' if new else 'SAME_TEST'
        else:
            reason = 'LOCAL_TOO_LENIENT?' 
            verdict=reason
        print(f'{ "AW" if ascore>d["score"] else "LW"} {task:32s} s{s} local={d["score"]} arena={ascore} | {verdict} | {fn}')
