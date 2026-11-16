"""Shared paths, layer selections and batch order for the H100 experiments."""
import os
from pathlib import Path
from typing import NamedTuple

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
STORAGE = Path(os.environ.get("SQZ_ROOT", REPO / "storage")).expanduser().resolve()
FRAMEWORK = Path(os.environ.get("LLAMAFACTORY_ROOT", REPO)).expanduser().resolve()

REFERENCE_LAYERS = tuple(i for i in range(32) if i % 8 != 7)
# Keep nested subsets so each comparison removes known checkpoint boundaries.
LAYERS = {
    28: REFERENCE_LAYERS,
    26: tuple(i for i in REFERENCE_LAYERS if i not in (11, 27)),
    20: tuple(i for i in REFERENCE_LAYERS if i % 4 != 3 and i % 8 != 2),
    24: tuple(i for i in REFERENCE_LAYERS if i % 4 != 3),
    0: (),
}


class BenchmarkProfile(NamedTuple):
    warmup: tuple[int, ...]
    timed: tuple[int, ...]
    longest: int

    @property
    def steps(self) -> list[int]:
        return [*self.warmup, *self.timed, self.longest]


PROFILES = {
    "baseline": BenchmarkProfile((28, 0, 24), (28, 24, 0, 0, 24, 28), 24),
    "20-vs-24": BenchmarkProfile((24, 20), (24, 20, 20, 24), 20),
}


def training_config() -> dict:
    import yaml

    cfg = yaml.safe_load((SOURCE / "reference/check10.yaml").read_text())
    # The recorded model, registered dataset and tokenized cache share a storage root.
    reference_storage = Path(cfg["model_name_or_path"]).parents[1]
    for key in ("model_name_or_path", "dataset_dir", "tokenized_path"):
        relative = Path(cfg[key]).relative_to(reference_storage)
        cfg[key] = str(STORAGE / relative)
    cfg["deepspeed"] = str(FRAMEWORK / "examples/deepspeed/ds_z3_config.json")
    return cfg
