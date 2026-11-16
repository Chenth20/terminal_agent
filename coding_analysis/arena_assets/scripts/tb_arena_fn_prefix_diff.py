import glob,re,sys,ast,json,difflib,textwrap
A='/storage/openpsi/users/chen/output'
def arena_blocks(task,fn):
    """all printed source blocks of fn from arena logs: (file, lines) ; pytest prints from def to the failing line"""
    out=[]
    for f in sorted(glob.glob(f'{A}/cth-tb21*/tasks/tb2-{task}/s*/tbench_test.log')):
        t=open(f,errors='ignore').read()
        if '= FAILURES =' not in t or fn not in t: continue
        blk=t.split('= FAILURES =')[1]; blk=re.split(r'\n=+ (?:warnings summary|PASSES|short test summary)',blk)[0]
        cur=None; buf=[]
        for l in blk.splitlines():
            m=re.match(r'^_+ (\w+)(?:\[.*\])? _+$',l)
            if m:
                if cur==fn and buf: out.append((f,buf)); buf=[]
                cur=m.group(1); continue
            if cur!=fn: continue
            if l.startswith('_ _ _') or l.startswith('----') or l.startswith('E ') or re.match(r'^/tests/',l) or re.match(r'^\.\./tests/',l):
                if buf: out.append((f,buf)); buf=[]
                cur=None; continue
            if l.startswith(('    ','>   ')): buf.append(l[4:])
        if cur==fn and buf: out.append((f,buf))
    return out
def local_fn(task,fn):
    for f in glob.glob(f'tasks/{task}/tests/*.py'):
        src=open(f).read(); tree=ast.parse(src); lines=src.splitlines()
        for n in ast.walk(tree):
            if isinstance(n,ast.FunctionDef) and n.name==fn:
                return n.lineno, lines[n.lineno-1:n.end_lineno]
    return None,[]
task,fn=sys.argv[1],sys.argv[2]
blocks=arena_blocks(task,fn)
if not blocks: print('no arena block'); sys.exit()
f,ab=max(blocks,key=lambda x:len(x[1]))   # the longest printed block = arena failed latest
ab=[l.rstrip() for l in ab]
l0,lb=local_fn(task,fn); lb=[l.rstrip() for l in lb]
print(f'### {task}::{fn}: arena longest block {len(ab)} lines (from {f.split("/output/")[1].split("/tasks/")[0]}); local fn {len(lb)} lines (starts line {l0})')
# compare arena block against the same-length prefix of local fn
pref=lb[:len(ab)]
d=list(difflib.unified_diff(pref,ab,'local(prefix)','arena',lineterm='',n=0))
if not d: print('   PREFIX IDENTICAL:', len(ab),'of',len(lb),'local lines covered')
else:
    for x in d[:40]: print('   '+x)
print(f'   -> local lines NOT visible in any arena log: {max(0,len(lb)-len(ab))} (lines {l0+len(ab)}..{l0+len(lb)-1})')
