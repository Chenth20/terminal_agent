"""Launch the SQZ single-node, eight-rank TH3K reproduction."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from h100_config import FRAMEWORK, LAYERS, SOURCE, STORAGE, training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, choices=sorted(LAYERS), default=28)
    parser.add_argument("--output", type=Path, required=True, help="New run directory on Ceph")
    parser.add_argument("--prepare-only", action="store_true", help="Write launch files without starting training")
    args = parser.parse_args()
    cfg = training_config()
    ds_config = Path(cfg["deepspeed"])
    for path in [Path(cfg["model_name_or_path"]) / "config.json",
                 Path(cfg["tokenized_path"]) / "manifest.json", ds_config]:
        if not path.is_file():
            parser.error(f"Missing training input: {path}")
    out = args.output.resolve()
    if not out.parent.is_dir():
        parser.error(f"Output parent does not exist: {out.parent}")
    if subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(out.parent)], text=True).strip() != "ceph":
        parser.error("Choose an output directory on the mounted Ceph filesystem")
    if not args.prepare_only:
        import torch
        if torch.cuda.device_count() != 8:
            parser.error("This schedule requires exactly eight visible GPUs")
    out.mkdir(exist_ok=False)
    for name in ["driver.py", "selective.py", "train.py"]:
        shutil.copyfile(SOURCE / "reference" / name, out / name)
    if args.layers != 28:
        (out / "train.py").rename(out / "train_reference.py")
        (out / "train.py").write_text(
            "import runpy\nfrom pathlib import Path\nimport selective\n"
            f"selective.OFFLOAD_LAYERS = frozenset({LAYERS[args.layers]!r})\n"
            "runpy.run_path(str(Path(__file__).with_name('train_reference.py')), run_name='__main__')\n"
        )
    cfg["output_dir"] = str(out / "checkpoints")
    (out / "check10.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(out, flush=True)
    if args.prepare_only:
        return
    cuda = STORAGE / "conda/envs/cuda124"
    env = os.environ.copy()
    env.update(PYTHONPATH=str(FRAMEWORK / "src"), CUDA_HOME=str(cuda),
               PATH=os.pathsep.join([str(Path(sys.executable).parent), str(cuda / "bin"), env["PATH"]]),
               HF_HOME=str(out / "hf-cache"), TRITON_CACHE_DIR=str(STORAGE / "runs/.triton-h100"),
               TORCH_EXTENSIONS_DIR=str(STORAGE / "runs/.torch-extensions-h100"),
               HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               IMAGE_PLACEHOLDER="<|lf_image_placeholder_unused|>",
               VIDEO_PLACEHOLDER="<|lf_video_placeholder_unused|>",
               AUDIO_PLACEHOLDER="<|lf_audio_placeholder_unused|>")
    info = [{"gpu": torch.cuda.get_device_name(i),
             "total_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(8)]
    (out / "hardware.json").write_text(json.dumps(info, indent=2))
    with (out / "driver.log").open("w") as log:
        subprocess.run([sys.executable, str(out / "driver.py")], cwd=FRAMEWORK, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
