"""Read LlamaFactory tokenized caches (arrow shards) and the perdh jsonl index without the `datasets` version
of the login node (which cannot open these caches). Shared by build_tree_sched.py and checks/."""

import json
import os

import numpy as np
import pyarrow as pa

IGNORE_INDEX = -100


def read_cache(path: str, columns: list[str]) -> dict[str, list[np.ndarray]]:
    """Read list columns of a saved-to-disk LF cache straight from its arrow shards (row order preserved)."""
    train_dir = os.path.join(path, "train")
    with open(os.path.join(train_dir, "state.json"), encoding="utf-8") as f:
        files = [d["filename"] for d in json.load(f)["_data_files"]]
    out: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    for fn in files:
        with pa.memory_map(os.path.join(train_dir, fn), "r") as src:
            table = pa.ipc.open_stream(src).read_all()
        for c in columns:
            for chunk in table.column(c).chunks:
                offsets = chunk.offsets.to_numpy()
                values = chunk.values.to_numpy(zero_copy_only=False)
                out[c].extend(values[offsets[i] : offsets[i + 1]].copy() for i in range(len(offsets) - 1))
    return out


def perdh_index(jsonl: str) -> list[tuple[int, int]]:
    """(traj_idx, step) per row of the perdh jsonl, read from the trailing fields without parsing contents."""
    idx = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            tail = line[line.rfind('"traj_idx"') :].rstrip().rstrip("}")
            d = json.loads("{" + tail + "}")
            idx.append((int(d["traj_idx"]), int(d["step"])))
    return idx
