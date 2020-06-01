from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy
import pytest
import torch
import yaml
from yaml import SafeLoader

from tests.utility import generate_data, get_data_directory
from yukarin_style.config import Config
from yukarin_style.generator import Generator


@pytest.fixture()
def train_config_path():
    return get_data_directory() / "train_config.yaml"


@pytest.fixture()
def style_transfer_path():
    return get_data_directory() / "style_transfer_100.npz"


@pytest.fixture()
def mapping_network_path():
    return get_data_directory() / "mapping_network_100.npz"


@pytest.fixture()
def style_encoder_path():
    return get_data_directory() / "style_encoder_100.npz"


def test_generator(
    train_config_path: Path,
    style_transfer_path: Path,
    mapping_network_path: Path,
    style_encoder_path: Path,
):
    with train_config_path.open() as f:
        d = yaml.load(f, SafeLoader)

    config = Config.from_dict(d)

    generator = Generator(
        config=config,
        style_transfer=style_transfer_path,
        mapping_network=mapping_network_path,
        style_encoder=style_encoder_path,
        use_gpu=True,
    )

    batch_size = 3

    x = torch.stack(
        [
            torch.from_numpy(
                generate_data(
                    wavelength=config.dataset.sampling_length // 2,
                    exponent=2 ** (i - 1),
                    amplitude=0.5,
                )[0]
            )
            for i in range(batch_size)
        ]
    )

    z = generator.generate_latent(batch_size=batch_size)
    s = generator.generate_style(x=None, z=z)
    y = generator.generate(x=x, s=s)

    for y_one in y:
        with NamedTemporaryFile(suffix=".npy") as f:
            pass
        numpy.save(f.name, y_one)
