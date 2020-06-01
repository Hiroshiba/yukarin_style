from pathlib import Path

from acoustic_feature_extractor.data.sampling_data import SamplingData
import numpy


def get_data_directory() -> Path:
    return Path(__file__).parent.relative_to(Path.cwd()) / "data"


def generate_data(
    wavelength: float, exponent: float, amplitude: float, length=300,
):
    random_phase = numpy.random.rand()
    wave = numpy.sin(
        (numpy.arange(length).astype(numpy.float32) / wavelength + random_phase)
        * 2
        * numpy.pi
    )
    wave = numpy.sign(wave) * numpy.abs(wave) ** exponent * amplitude
    feature = numpy.stack([wave, wave]).T
    silence = numpy.zeros_like(wave, dtype=bool)
    return feature, silence


def generate_and_save_data(
    feature_dir: Path,
    silence_dir: Path,
    wavelength: float,
    exponent: float,
    amplitude: float,
    length=300,
):
    feature, silence = generate_data(
        wavelength=wavelength, exponent=exponent, amplitude=amplitude, length=length,
    )

    filename = f"{wavelength}_{exponent}_{amplitude}.npy"
    SamplingData(array=feature, rate=100).save(feature_dir / filename)
    SamplingData(array=silence, rate=100).save(silence_dir / filename)
