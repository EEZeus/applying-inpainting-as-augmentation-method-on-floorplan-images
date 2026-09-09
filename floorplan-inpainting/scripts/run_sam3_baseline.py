#!/usr/bin/env python3
"""Run Meta's official SAM 3 trainer with this project's external Hydra config."""

import argparse
import os
import random
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from sam3.train.train import single_node_runner
from sam3.train.utils.train_utils import register_omegaconf_resolvers


ROOT = Path(__file__).resolve().parents[1]


def _enable_trainable_vit_mlp() -> None:
    """Replace SAM3's inference-only fused ViT MLP path during fine-tuning."""
    from sam3.model.vitdet import Mlp

    def trainable_forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)

    Mlp.forward = trainable_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sam3_floorplan.yaml")
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--overrides", nargs="*", default=[],
                        help="Hydra dotlist overrides, e.g. "
                             "paths.dataset_root=/abs/path paths.experiment_log_dir=/abs/path")
    args = parser.parse_args()

    if args.mode == "train":
        _enable_trainable_vit_mlp()

    register_omegaconf_resolvers()
    config = args.config.resolve()
    with initialize_config_dir(version_base="1.2", config_dir=str(config.parent)):
        cfg = compose(config_name=config.stem, overrides=list(args.overrides))

    with open_dict(cfg):
        if args.mode == "test":
            checkpoint = Path(cfg.trainer.checkpoint.save_dir) / "checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Trained SAM 3 checkpoint not found: {checkpoint}")
            cfg.trainer.mode = "val"
            cfg.paths.eval_split = "test"
            # The complete fine-tuned state is restored by Trainer from checkpoint.pt.
            cfg.trainer.model.load_from_HF = False
            cfg.trainer.model.checkpoint_path = None
        else:
            local_checkpoint = os.environ.get("SAM3_CHECKPOINT")
            if local_checkpoint:
                checkpoint = Path(local_checkpoint).resolve()
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"SAM3_CHECKPOINT does not exist: {checkpoint}")
                cfg.trainer.model.checkpoint_path = str(checkpoint)
                cfg.trainer.model.load_from_HF = False

    single_node_runner(cfg, random.randint(10000, 65000))


if __name__ == "__main__":
    main()
