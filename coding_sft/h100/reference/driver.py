from pathlib import Path
import subprocess,sys,json
w=Path(__file__).parent
with (w/'check10.log').open('w') as f:
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',str(w/'train.py'),'check10'],stdout=f,stderr=subprocess.STDOUT,check=True)
for rank in range(8):
    rows=[json.loads(l) for l in (w/f'check10-rank{rank}.jsonl').read_text().splitlines()]
    assert [v['step'] for v in rows]==list(range(1,11)) and all(v['passed'] for v in rows)
ck=w/'checkpoints/checkpoint-10'
assert (ck/'trainer_state.json').is_file() and len(list(ck.rglob('*optim_states.pt')))==8
print('check10 passed',flush=True)
