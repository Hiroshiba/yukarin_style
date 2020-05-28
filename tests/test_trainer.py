from pathlib import Path
from yukarin_style.trainer import create_trainer

import pytest
import yaml
from yaml import SafeLoader

from tests.utility import get_data_directory
from yukarin_style.config import Config

import numpy
from tempfile import TemporaryDirectory

from acoustic_feature_extractor.data.sampling_data import SamplingData


@pytest.fixture()
def train_config_path():
    return get_data_directory() / "train_config.yaml"


def generate_data(feature_dir: Path, silence_dir: Path, wavelength: float, length=300):
    random_phase = numpy.random.rand()
    wave = numpy.sin(
        (numpy.arange(length).astype(numpy.float32) / wavelength + random_phase)
        * 2
        * numpy.pi
    )

    feature = numpy.stack([wave, wave]).T
    SamplingData(array=feature, rate=100).save(feature_dir / f"{wavelength}.npy")

    silence = numpy.zeros_like(wave, dtype=bool)
    SamplingData(array=silence, rate=100).save(silence_dir / f"{wavelength}.npy")


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

    for wavelength in numpy.linspace(
        start=10, stop=config.dataset.sampling_length, num=300
    ):
        generate_data(
            feature_dir=feature_dir, silence_dir=silence_dir, wavelength=wavelength,
        )

    # train
    with TemporaryDirectory() as output_dir:
        pass

    trainer = create_trainer(config_dict=config.to_dict(), output=Path(output_dir))
    trainer.run()
