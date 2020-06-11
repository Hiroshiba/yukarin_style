import pytest
from acoustic_feature_extractor.data.sampling_data import SamplingData
from yukarin_style.dataset import extract_input
import numpy


@pytest.mark.parametrize(
    "sampling_length,data_length,padding_length",
    [(16, 16, 0), (16, 16, 8), (16, 8, 8), (8, 16, 8), (8, 16, 0)],
)
def test_extract_input(sampling_length: int, data_length: int, padding_length: int):
    silence_data = SamplingData(array=numpy.zeros(data_length, dtype=bool), rate=1)
    spectrogram_data = SamplingData(
        array=numpy.linspace(start=1, stop=2, num=data_length)[:, numpy.newaxis],
        rate=1,
    )
    for _ in range(100):
        spectrogram = extract_input(
            sampling_length=sampling_length,
            spectrogram_data=spectrogram_data,
            silence_data=silence_data,
            min_not_silence_length=min(sampling_length, data_length),
            padding_length=padding_length,
            padding_value=numpy.nan,
        )["spectrogram"]

        assert len(spectrogram) == sampling_length + padding_length * 2

        if sampling_length <= data_length:
            assert numpy.isnan(spectrogram).sum() <= padding_length * 2
        else:
            assert (
                numpy.isnan(spectrogram).sum()
                == sampling_length - data_length + padding_length * 2
            )

        if padding_length == 0:
            data = spectrogram
        else:
            data = spectrogram[padding_length:-padding_length]
        assert (~numpy.isnan(data)).sum() >= min(sampling_length, data_length)
