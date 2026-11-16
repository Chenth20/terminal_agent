#!/usr/bin/env python3
"""Training entry for arm B (trunk+branches tree attention).

LlamaFactory's SFT workflow builds its own collator, so this entry:
  1. applies the class-level model patch (coding_sft/patches/tree_attn.py),
  2. replaces the collator class used by llamafactory.train.sft.workflow with the tree collator,
  3. hands over to llamafactory's normal CLI path (get_train_args / trainer / DeepSpeed ZeRO-3).

Run under torchrun exactly like `llamafactory-cli train` would:

    torchrun --nproc_per_node 8 --nnodes N --node_rank R --master_addr A --master_port P \
        coding_sft/patches/train_tree.py <train.yaml> [k=v ...]

Every dataset row must carry the tree columns (input_ids/labels/position_ids/tree_forks/
tree_branch_lens/trunk_len), i.e. tokenized_path points at a cache from prepare_tree*.py.
"""

import os
import sys


for _k in ("IMAGE", "VIDEO", "AUDIO"):
    os.environ.setdefault(f"{_k}_PLACEHOLDER", f"<|lf_{_k.lower()}_placeholder_unused|>")

# Per-rank compiled-kernel cache on shared storage (set by the launchers as TREE_KCACHE_BASE): must happen before
# torch/triton are imported. One directory per global rank -- a single directory shared by 16 ranks on 2 nodes
# raced on mkdir / os.replace(tmp.pid_*) / dlopen(__triton_launcher.so) and died with FileNotFoundError or
# ESTALE on CPFS. The same rank re-uses its own directory after a pre-emption restart (no re-autotune).
#
# TREE_KCACHE_SEED (optional): a per-DATASET seed cache shared by every run on that dataset, since the set of
# compiled flex graphs depends only on the data's T_pad buckets (+T_MULT), not on the run name.  Safe sharing:
#   * pull  (all ranks, at startup): copy seed -> own rank dir, files that are missing locally only; a live
#     process never opens a seed file directly, so nothing we dlopen can be replaced under us;
#   * publish (global rank 0 only, background thread): copy own rank dir -> seed, new files only, each written
#     to a temp name and os.replace'd into place (atomic on CPFS), never overwriting an existing file.
# Single writer + immutable files + readers working on private copies = no rename/dlopen races.
_KCACHE = os.environ.get("TREE_KCACHE_BASE")
_KSEED = os.environ.get("TREE_KCACHE_SEED")
_RANK = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
if _KCACHE:
    _rank_dir = os.path.join(_KCACHE, f"rank{_RANK}")
    for _sub in ("triton", "inductor"):
        os.makedirs(os.path.join(_rank_dir, _sub), exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = os.path.join(_rank_dir, "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(_rank_dir, "inductor")


def _kcache_sync(src: str, dst: str, log: bool) -> int:
    """Copy files missing in dst from src (atomic per file, never overwrite). Returns #copied."""
    import shutil
    import time

    n = 0
    t0 = time.time()
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_dir = os.path.join(dst, rel) if rel != "." else dst
        made = False
        for fn in files:
            if fn.startswith("tmp.") or fn.endswith((".lock", ".tmp")):
                continue
            s_path = os.path.join(root, fn)
            d_path = os.path.join(out_dir, fn)
            if os.path.exists(d_path):
                continue
            if not made:
                os.makedirs(out_dir, exist_ok=True)
                made = True
            tmp = f"{d_path}.tmp.{os.getpid()}"
            try:
                shutil.copyfile(s_path, tmp)
                os.replace(tmp, d_path)
                n += 1
            except OSError:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    if log and n:
        print(f"[tree] kcache: {n} files {src} -> {dst} in {time.time() - t0:.0f}s", flush=True)
    return n


if _KCACHE and _KSEED:
    for _sub in ("triton", "inductor"):
        _s = os.path.join(_KSEED, _sub)
        if os.path.isdir(_s):
            _kcache_sync(_s, os.path.join(_rank_dir, _sub), log=_RANK == "0")
    if _RANK == "0":
        import threading

        def _publish_loop() -> None:
            import time

            while True:
                time.sleep(600)
                for sub in ("triton", "inductor"):
                    os.makedirs(os.path.join(_KSEED, sub), exist_ok=True)
                    _kcache_sync(os.path.join(_rank_dir, sub), os.path.join(_KSEED, sub), log=True)

        threading.Thread(target=_publish_loop, name="kcache-publish", daemon=True).start()
        print(f"[tree] kcache seed {_KSEED}: pulled at startup, rank0 publishes every 10 min", flush=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from tree_attn import TreeDataCollator, apply_tree_patch, reassert_tree_patch  # noqa: E402

PAD_TOKEN_ID = int(os.environ.get("TREE_PAD_TOKEN_ID", "248044"))


class TreeCollatorFactory:
    """Drop-in replacement for SFTDataCollatorWith4DAttentionMask: ignores LF's kwargs, emits tree batches."""

    def __init__(self, *args, **kwargs) -> None:
        self._inner = TreeDataCollator(pad_token_id=PAD_TOKEN_ID)
        self.template = kwargs.get("template")
        self.tokenizer = kwargs.get("tokenizer")

    def __call__(self, features):
        return self._inner(features)


def _install_after_llamafactory_patch() -> None:
    """llamafactory's patch_model() (model/patcher.py, called by model/loader.py at load time, i.e. AFTER our
    apply_tree_patch()) assigns its own cu_seqlens forward to Qwen3_5GatedDeltaNet -- which silently turned the
    24 linear layers back into a plain causal scan (dbg step-1 loss 1.609 vs 1.410). Wrap the name loader.py
    actually calls, re-install the tree forwards right after, and fail loudly if they are not the live ones."""
    import llamafactory.model.loader as loader

    original = loader.patch_model

    def patch_model_then_tree(*args, **kwargs):
        out = original(*args, **kwargs)
        reassert_tree_patch()
        import transformers.models.qwen3_5.modeling_qwen3_5 as M

        from tree_attn import tree_attn_forward, tree_gdn_forward, tree_text_forward

        live = (
            M.Qwen3_5TextModel.forward is tree_text_forward,
            M.Qwen3_5Attention.forward is tree_attn_forward,
            M.Qwen3_5GatedDeltaNet.forward is tree_gdn_forward,
        )
        if not all(live):
            raise RuntimeError(f"tree forwards are not installed after patch_model: {live}")
        return out

    loader.patch_model = patch_model_then_tree


def main() -> None:
    import llamafactory.train.sft.workflow as workflow

    apply_tree_patch()
    _install_after_llamafactory_patch()
    workflow.SFTDataCollatorWith4DAttentionMask = TreeCollatorFactory
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[tree] training entry: patch applied, collator replaced (pad_token_id={PAD_TOKEN_ID})", flush=True)

    from llamafactory.train.tuner import run_exp

    run_exp()


if __name__ == "__main__":
    main()