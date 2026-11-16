import re,glob,os,collections,ast
A='/storage/openpsi/users/chen/output'
jobs=[d for d in os.listdir(A) if d.startswith('cth-tb21')]
# local test functions: name -> normalized body lines
def local_funcs(task):
    out={}
    for f in glob.glob(f'tasks/{task}/tests/*.py'):
        src=open(f,errors='ignore').read()
        try: tree=ast.parse(src)
        except Exception: continue
        lines=src.splitlines()
        for n in ast.walk(tree):
            if isinstance(n,ast.FunctionDef):
                body=lines[n.lineno-1:n.end_lineno]
                out[n.name]=set(l.strip() for l in body if l.strip())
    return out
# arena: parse FAILURES block -> per function the printed source lines (prefixed by 4 spaces or '>   ')
def arena_funcs(txt):
    if '= FAILURES =' not in txt: return {}
    blk=txt.split('= FAILURES =')[1]
    blk=re.split(r'\n=+ (?:warnings summary|PASSES|short test summary)',blk)[0]
    funcs={}; cur=None
    for l in blk.splitlines():
        m=re.match(r'^_+ (\w+)(?:\[.*\])? _+$',l)
        if m: cur=m.group(1); funcs.setdefault(cur,set()); continue
        if cur is None: continue
        if l.startswith('/tests/') or l.startswith('../tests/'): continue
        if l.startswith('    ') or l.startswith('>   '):
            s=l[4:].rstrip()
            if s.strip() and not s.startswith(('E ',)):
                funcs[cur].add(s.strip())
        elif l.startswith('_ _ _'): cur=None   # into library frames; stop
    return funcs
report={}
for task in sorted(os.listdir('tasks')):
    L=local_funcs(task)
    if not L: continue
    diffs=collections.defaultdict(set); scanned=0
    for j in jobs:
        for f in glob.glob(f'{A}/{j}/tasks/tb2-{task}/s*/tbench_test.log'):
            try: t=open(f,errors='ignore').read()
            except: continue
            F=arena_funcs(t)
            if not F: continue
            scanned+=1
            for name,lines in F.items():
                if name not in L:
                    diffs[name].add('<<function missing locally>>'); continue
                for s in lines:
                    if s not in L[name] and not s.startswith(('"""',"'''",'#','@')):
                        diffs[name].add(s)
    if diffs: report[task]=(scanned,diffs)
for task,(n,d) in report.items():
    print(f'== {task}  (arena failure logs scanned: {n})')
    for fn,ls in d.items():
        print(f'   {fn}:')
        for s in sorted(ls)[:6]: print(f'      | {s[:120]}')
print(len(report),'tasks with test-source differences')
