"""Measure activation offload on the same eight-rank training batch."""
import argparse
import json
from pathlib import Path
import sys
import time

import yaml

from h100_config import LAYERS, PROFILES, SOURCE, training_config


def prepare(out, profile):
    modes = PROFILES[profile].steps
    from datasets import DatasetDict, load_from_disk
    cfg = training_config()
    schedule = Path(cfg["tokenized_path"])
    lengths = json.loads((schedule / "manifest.json").read_text())["traj_lengths"]
    flat = load_from_disk(str(schedule.parent.parent / "flat"))["train"]
    medium = min(range(len(lengths)), key=lambda i: abs(lengths[i] - 65536))
    longest = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)[:8]
    indices = [medium] * (8 * (len(modes) - 1)) + longest
    expected = [sum((lengths[i] + 7) // 8 * 8 for i in indices[k:k+8])
                for k in range(0, len(indices), 8)]
    DatasetDict({"train": flat.select(indices)}).save_to_disk(str(out / "inputs"))
    (out / "inputs.json").write_text(json.dumps(dict(profile=profile, modes=modes, indices=indices, expected=expected,
                    medium=lengths[medium], longest=[lengths[i] for i in longest])))
    cfg.update(tokenized_path=str(out / "inputs"), output_dir=str(out / "scratch"),
               max_steps=len(modes), gradient_accumulation_steps=1,
               save_strategy="no", learning_rate=0.0, plot_loss=False)
    (out / "bench.yaml").write_text(yaml.safe_dump(cfg))


def measure(out):
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp
    sys.path.insert(0, str(SOURCE / "reference"))
    import selective

    dist = torch.distributed
    spec = json.loads((out / "inputs.json").read_text())
    expected = spec["expected"]
    modes = spec.get("modes", PROFILES[spec.get("profile", "baseline")].steps)

    class Done(Exception):
        pass

    class Timing(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            self.step = state.global_step
            selective.OFFLOAD_LAYERS = frozenset(LAYERS[modes[self.step]])
            selective.moved.clear()
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            self.start_tokens = state.num_input_tokens_seen
            self.start = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            seconds = time.perf_counter() - self.start
            stats = torch.tensor([seconds, torch.cuda.max_memory_allocated()/2**30,
                                  torch.cuda.max_memory_reserved()/2**30], device=args.device,
                                 dtype=torch.float64)
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            tokens = state.num_input_tokens_seen - self.start_tokens
            if tokens != expected[self.step] or selective.moved != selective.OFFLOAD_LAYERS:
                raise RuntimeError(f"Step {self.step+1} mismatch: {tokens=} {selective.moved=}")
            if dist.get_rank() == 0:
                sec, alloc, reserved = stats.tolist()
                row = dict(step=self.step+1, layers=modes[self.step], tokens=tokens,
                           seconds=sec, tokens_per_second=tokens/sec,
                           allocated_GiB=alloc, reserved_GiB=reserved)
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
            if self.step == len(modes)-1:
                dist.barrier()
                raise Done()

    try:
        run_exp(yaml.safe_load((out / "bench.yaml").read_text()), callbacks=[Timing()])
    except Done:
        return
    raise RuntimeError("Benchmark did not reach its last batch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "measure"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="baseline")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.directory, args.profile)
    else:
        measure(args.directory)
