import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy
import torch
import yaml
from pytorch_trainer.dataset.convert import concat_examples
from tqdm import tqdm

from utility.save_arguments import save_arguments
from yukarin_style.config import Config
from yukarin_style.dataset import create_dataset
from yukarin_style.generator import Generator


def _extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def _get_network_paths(
    model_dir: Path,
    iteration: int = None,
    style_transfer_prefix="style_transfer_",
    mapping_network_prefix="mapping_network_",
    style_encoder_prefix="style_encoder_",
):
    if iteration is None:
        style_transfer_path = sorted(
            model_dir.glob(f"{style_transfer_prefix}*.pth"), key=_extract_number
        )[-1]
        mapping_network_path = sorted(
            model_dir.glob(f"{mapping_network_prefix}*.pth"), key=_extract_number
        )[-1]
        style_encoder_path = sorted(
            model_dir.glob(f"{style_encoder_prefix}*.pth"), key=_extract_number
        )[-1]
    else:
        style_transfer_path = model_dir / f"{style_transfer_prefix}{iteration}.pth"
        mapping_network_path = model_dir / f"{mapping_network_prefix}{iteration}.pth"
        style_encoder_path = model_dir / f"{style_encoder_prefix}{iteration}.pth"
    return style_transfer_path, mapping_network_path, style_encoder_path


def generate(
    model_dir: Path,
    model_iteration: Optional[int],
    model_config: Optional[Path],
    output_dir: Path,
    use_gpu: bool,
    style_num: int,
    content_num: int,
):
    if model_config is None:
        model_config = model_dir / "config.yaml"

    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / "arguments.yaml", generate, locals())

    config = Config.from_dict(yaml.safe_load(model_config.open()))
    sampling_length = config.dataset.sampling_length
    padding_length = config.dataset.padding_length

    style_transfer_path, mapping_network_path, style_encoder_path = _get_network_paths(
        model_dir=model_dir, iteration=model_iteration,
    )
    generator = Generator(
        config=config,
        style_transfer=style_transfer_path,
        mapping_network=mapping_network_path,
        style_encoder=style_encoder_path,
        use_gpu=use_gpu,
    )

    dataset = create_dataset(config.dataset)["eval"]

    batch = concat_examples([dataset[i] for i in range(max(style_num, content_num))])

    for i_x, x in tqdm(enumerate(batch["x"][:content_num].split(1)), desc="generate"):
        x_ref = batch["x"][:style_num, sampling_length:-sampling_length]
        z = batch["z1"][:style_num]

        x = x[:, padding_length:-padding_length]
        x = x.expand(style_num, x.shape[1], x.shape[2])

        s = generator.generate_style(x=x_ref, z=None)
        outputs = generator.generate(x=x, s=s)
        for i_style, output in enumerate(outputs):
            numpy.save(Path(output_dir, f"output-ref{i_style}-{i_x}.npy"), output)

        s = generator.generate_style(x=None, z=z)
        outputs = generator.generate(x=x, s=s)
        for i_style, output in enumerate(outputs):
            numpy.save(Path(output_dir, f"output-latent{i_style}-{i_x}.npy"), output)

        numpy.save(
            Path(output_dir, f"input-{i_x}.npy"),
            x[0][padding_length:-padding_length].numpy(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--model_iteration", type=int)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--style_num", default=4, type=int)
    parser.add_argument("--content_num", default=4, type=int)
    generate(**vars(parser.parse_args()))
