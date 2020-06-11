from pathlib import Path
from tempfile import TemporaryDirectory

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
    return get_data_directory() / "style_transfer_50000.pth"


@pytest.fixture()
def mapping_network_path():
    return get_data_directory() / "mapping_network_50000.pth"


@pytest.fixture()
def style_encoder_path():
    return get_data_directory() / "style_encoder_50000.pth"


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
    sampling_length = config.dataset.sampling_length

    x = numpy.stack(
        [
            generate_data(
                wavelength=sampling_length // 2, exponent=2 ** (i - 1), amplitude=0.5,
            )[0]
            for i in range(batch_size)
        ]
    )

    x_torch = torch.from_numpy(x)

    z = generator.generate_latent(batch_size=1)
    z = z.repeat([batch_size, 1])

    s_latent = generator.generate_style(x=None, z=z)
    y_latent = generator.generate(x=x_torch, s=s_latent)

    s_ref = generator.generate_style(x=x_torch[:, :sampling_length], z=None)
    y_ref = generator.generate(x=x_torch, s=s_ref)

    with TemporaryDirectory() as output_dir:
        pass

    Path(output_dir).mkdir()

    for i, (x_one, y_one_latent, y_one_ref) in enumerate(zip(x, y_latent, y_ref)):
        numpy.save(Path(output_dir, f"input-{i}.npy"), x_one)
        numpy.save(Path(output_dir, f"generate-latent-{i}.npy"), y_one_latent)
        numpy.save(Path(output_dir, f"generate-ref-{i}.npy"), y_one_ref)

    print(f"generated data dir: {output_dir}")
