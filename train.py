import argparse
from pathlib import Path
from typing import Optional

import yaml

from yukarin_style.trainer import create_trainer


def train(config_yaml_path: Path, output: Path, dataset_dir: Optional[Path]):
    with config_yaml_path.open() as f:
        d = yaml.safe_load(f)

    trainer = create_trainer(config_dict=d, output=output, dataset_dir=dataset_dir)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset_dir", type=Path)
    train(**vars(parser.parse_args()))
