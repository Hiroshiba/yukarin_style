from pathlib import Path
from tempfile import TemporaryDirectory

import numpy
import pytest
import yaml
from tests.utility import generate_and_save_data, get_data_directory
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
    config.dataset.spectrogram_filelist = Path(
        tmp_dir.name, config.dataset.spectrogram_filelist.relative_to("/tmp")
    )
    config.dataset.silence_filelist = Path(
        tmp_dir.name, config.dataset.silence_filelist.relative_to("/tmp")
    )

    feature_dir = Path(tmp_dir.name, "feature")
    feature_dir.mkdir()
    silence_dir = Path(tmp_dir.name, "silence")
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

    config.dataset.spectrogram_filelist.write_text(
        "\n".join(map(str, feature_dir.glob("*.npy")))
    )
    config.dataset.silence_filelist.write_text(
        "\n".join(map(str, silence_dir.glob("*.npy")))
    )

    # train
    with TemporaryDirectory() as output_dir:
        pass

    trainer = create_trainer(
        config_dict=config.to_dict(), output=Path(output_dir), dataset_dir=None
    )
    trainer.run()
