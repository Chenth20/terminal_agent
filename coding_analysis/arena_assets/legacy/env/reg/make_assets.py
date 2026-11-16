#!/usr/bin/env python3
"""Build Env spec assets for selected TerminalHorizon tasks.

For each task directory under data/tasks:
  assets/<task>/instruction.md            -> Env.query (uploaded file)
  assets/<task>/oracle/tests/*            -> Env.oracle (terminal-bench.2.0.v1 layout)
  assets/<task>/oracle/run-tests.sh       -> runs official tests/test.sh
  spec/<task>.json                        -> Env spec with placeholders for URIs
"""
import json, os, re, shutil, sys
try:
    import tomllib
except ModuleNotFoundError:  # python 3.10 on this cluster
    import tomli as tomllib
from pathlib import Path

CA_ROOT = Path(__file__).resolve().parents[2]          # coding_arena/
WORK = Path(os.environ.get("TH_WORK", CA_ROOT / "work"))
SPECS = WORK / "env-specs"
ASSETS = Path(os.environ.get("TH_ASSETS", "/storage/openpsi/users/chen/code_data/terminal_horizon/env_assets"))
TASKS_ROOT = Path(os.environ.get("TH_TASKS_ROOT", "/storage/openpsi/users/chen/code_data/terminal_horizon/data/tasks"))
_imf = Path(os.environ.get("IMAGES_FILE", "images.json"))
IMAGES = json.loads((_imf if _imf.is_absolute() else WORK / "manifests" / _imf).read_text())
BATCH_TAG = os.environ.get("BATCH_TAG", "th-probe5-r1")
IMAGE_PREFIX = "reg.antgroup-inc.cn/areal_agent/terminalhorizon"
# Optional per-task metadata from the 10-25 server handoff (work/manifests/handoff_meta.json):
# digest / flat / exec_prefix / usage. flat images need a wait for /opt/flatten/ready.json before the agent
# starts; exec_prefix images need verifier commands wrapped by /opt/flatten/main_exec.py.
_mf = os.environ.get("META_FILE")
META = json.loads((Path(_mf) if Path(_mf).is_absolute() else WORK / "manifests" / _mf).read_text()) if _mf else {}

FLAT_READY_HOOK = (
    "for i in $(seq 1 120); do [ -f /opt/flatten/ready.json ] && break; sleep 5; done; "
    "echo \"[th-setup] flatten ready: $(test -f /opt/flatten/ready.json && echo yes || echo NO)\"; "
)

ENVD_HOOK = (
    "mkdir -p /home/admin/logs; "
    "if ! grep -q ':2CBC 00000000:0000 0A' /proc/net/tcp 2>/dev/null; then "
    "  (setsid nohup /home/admin/bin/arca_envd serve --addr :11452 >>/home/admin/logs/arca_envd.stdout.log 2>&1 </dev/null &); sleep 3; "
    "fi; "
    "echo \"[th-setup] envd listen: $(grep -c ':2CBC 00000000:0000 0A' /proc/net/tcp)\"; exit 0"
)

RUN_TESTS = """#!/bin/sh
# terminal-bench 2.0 oracle entrypoint: run the task's official test script.
set -u
mkdir -p /logs/verifier
if [ -f /tests/run-tests.sh ] && [ ! -f /tests/test.sh ]; then
  echo "missing /tests/test.sh" >&2
  exit 0
fi
sh /tests/test.sh
echo "oracle finished; reward=$(cat /logs/verifier/reward.txt 2>/dev/null)" >&2
"""

# exec_prefix images (handoff usage: "agent 和 verifier 命令统一加前缀 python3 /opt/flatten/main_exec.py;
# 注入测试后先运行 --prepare-injected"). Only the verifier side can be wrapped here.
RUN_TESTS_PREFIXED = RUN_TESTS.replace(
    "sh /tests/test.sh",
    "python3 /opt/flatten/main_exec.py --prepare-injected\npython3 /opt/flatten/main_exec.py sh /tests/test.sh",
)

def main(tasks):
    for task in tasks:
        src = TASKS_ROOT / task
        if not src.is_dir():
            sys.exit(f"missing task dir: {src}")
        if task not in IMAGES:
            sys.exit(f"no image recorded for {task} in images.json")
        dst = ASSETS / task
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "oracle").mkdir(parents=True)
        shutil.copy2(src / "instruction.md", dst / "instruction.md")
        shutil.copytree(src / "tests", dst / "oracle" / "tests")
        assert not (dst / "solution").exists() and "solution" not in {p.name for p in (dst / "oracle").iterdir()}
        meta = META.get(task, {})
        rt = dst / "oracle" / "run-tests.sh"
        rt.write_text(RUN_TESTS_PREFIXED if meta.get("exec_prefix") else RUN_TESTS)
        rt.chmod(0o755)
        cfg = tomllib.loads((src / "task.toml").read_text())
        env, agent, verifier = cfg["environment"], cfg["agent"], cfg["verifier"]
        image = IMAGES[task]
        spec = {
            "query": {"uri": "PLACEHOLDER", "media_type": "text/markdown"},
            "oracle": {"uri": "PLACEHOLDER", "media_type": "application/vnd.envarena.oracle.directory"},
            "oracle_schema": "terminal-bench.2.0.v1",
            "runtime": {"kind": "image", "image": image, "profile": "image_networked"},
            "timeout_seconds": int(env["build_timeout_sec"] + agent["timeout_sec"] + verifier["timeout_sec"] + 600),
            "workdir": env["workdir"],
            "timeouts": {
                "setup_seconds": min(int(env["build_timeout_sec"]), 900),
                "agent_seconds": int(agent["timeout_sec"]),
                "collect_seconds": 300,
            },
            "resource_hints": {
                "cpu": str(env["cpus"]),
                "memory": f'{env["memory_mb"]}Mi',
                "disk": f'{env["storage_mb"]}Mi',
            },
            # Debian 13 (trixie) images: ARCA's installer nohup-starts arca_envd before the binary is
            # fully written ("No such file or directory"), so :11452 never listens and shared_env
            # reward fails with 502. Restart it in setup if it is not listening.
            "hooks": {"setup_script": (FLAT_READY_HOOK if meta.get("flat") else "") + ENVD_HOOK},
            "tags": ["source:terminal-horizon", "domain:terminal", "chen", BATCH_TAG],
            "metadata": {
                "task_id": task,
                "image": image,
                "category": cfg["metadata"].get("category", ""),
                "subcategory": cfg["metadata"].get("subcategory", ""),
                "allow_internet": str(env.get("allow_internet", "")),
                **({"image_digest": meta["digest"], "handoff_category": meta.get("category", ""),
                    "handoff_usage": meta.get("usage", ""),
                    "original_single_service": meta.get("original_single_service", "")} if meta else {}),
            },
        }
        p = SPECS / f"{task}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n")
        print("built", task)

if __name__ == "__main__":
    main(sys.argv[1:])
