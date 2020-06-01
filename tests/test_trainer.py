from pathlib import Path
from tempfile import TemporaryDirectory

import numpy
import pytest
import yaml
from tests.utility import get_data_directory, generate_and_save_data
from yaml import SafeLoader

from yukarin_style.config import Config
from yukarin_style.trainer import create_trainer


@pytest.fixture()
def train_config_path():
    return get_data_directory() / "train_config.yaml"


def test_trainer(train_config_path: Path):
    with train_config_path.open() as f:
        d = yaml.load(f, SafeLoader)

    config = Config.from_dict(d)

    # dummy data
    tmp_dir = TemporaryDirectory()
    config.dataset.spectrogram_glob = config.dataset.spectrogram_glob.replace(
        "/tmp", tmp_dir.name
    )
    config.dataset.silence_glob = config.dataset.silence_glob.replace(
        "/tmp", tmp_dir.name
    )

    feature_dir = Path(config.dataset.spectrogram_glob).parent
    feature_dir.mkdir()
    silence_dir = Path(config.dataset.silence_glob).parent
    silence_dir.mkdir()

    for exponent in numpy.logspace(-2, 1, num=10):
        for amplitude in numpy.linspace(0.1, 1, num=10):
            generate_and_save_data(
                feature_dir=feature_dir,
                silence_dir=silence_dir,
                wavelength=config.dataset.sampling_length // 2,
                exponent=exponent,
                amplitude=amplitude,
            )

    # train
    with TemporaryDirectory() as output_dir:
        pass

    trainer = create_trainer(config_dict=config.to_dict(), output=Path(output_dir))
    trainer.run()
