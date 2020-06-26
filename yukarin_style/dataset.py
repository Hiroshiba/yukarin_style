import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy
from acoustic_feature_extractor.data.sampling_data import SamplingData
from temp_cache import TempCache
from torch.utils.data.dataset import ConcatDataset, Dataset

from yukarin_style.config import DatasetConfig
from yukarin_style.utility.dataset_utility import default_convert


@dataclass
class InputData:
    spectrogram: SamplingData
    silence: SamplingData


@dataclass
class LazyInputData:
    spectrogram_path: Path
    silence_path: Path

    def generate(self):
        return InputData(
            spectrogram=SamplingData.load(str(TempCache(self.spectrogram_path))),
            silence=SamplingData.load(str(TempCache(self.silence_path))),
        )


def extract_input(
    sampling_length: int,
    spectrogram_data: SamplingData,
    silence_data: SamplingData,
    min_not_silence_length: int,
    padding_length: int,
    padding_value=0,
):
    """
    :return:
        spectrogram: (sampling_length, ?)
        silence: (sampling_length, )
    """
    assert spectrogram_data.rate == silence_data.rate

    spectrogram = spectrogram_data.array

    length = len(spectrogram)
    if length < sampling_length:
        p_start = numpy.random.randint(sampling_length - length + 1)
        p_end = sampling_length - length - p_start
        spectrogram = numpy.pad(
            spectrogram,
            [[p_start, p_end], [0, 0]],
            mode="constant",
            constant_values=padding_value,
        )
        length = sampling_length

    for _ in range(10000):
        if length > sampling_length:
            offset = numpy.random.randint(length - sampling_length)
        else:
            offset = 0

        silence = numpy.squeeze(silence_data.array[offset : offset + sampling_length])
        if (~silence).sum() >= min_not_silence_length:
            break
    else:
        raise Exception("cannot pick not silence data")

    start, end = offset - padding_length, offset + sampling_length + padding_length
    if start < 0 or end > length:
        shape = list(spectrogram.shape)
        shape[0] = sampling_length + padding_length * 2
        new_spectrogram = (
            numpy.ones(shape=shape, dtype=spectrogram.dtype) * padding_value
        )
        if start < 0:
            p_start = -start
            start = 0
        else:
            p_start = 0
        if end > length:
            p_end = sampling_length + padding_length * 2 - (end - length)
            end = length
        else:
            p_end = sampling_length + padding_length * 2
        new_spectrogram[p_start:p_end] = spectrogram[start:end]
        spectrogram = new_spectrogram
    else:
        spectrogram = spectrogram[start:end]

    return dict(spectrogram=spectrogram, silence=silence)


def generate_latent(latent_size: int):
    return numpy.random.randn(latent_size).astype(numpy.float32)


class BaseSpectrogramDataset(Dataset):
    def __init__(
        self, sampling_length: int, min_not_silence_length: int, padding_length: int
    ):
        self.sampling_length = sampling_length
        self.min_not_silence_length = min_not_silence_length
        self.padding_length = padding_length

    def make_input(
        self, spectrogram_data: SamplingData, silence_data: SamplingData,
    ):
        return extract_input(
            sampling_length=self.sampling_length,
            spectrogram_data=spectrogram_data,
            silence_data=silence_data,
            min_not_silence_length=self.min_not_silence_length,
            padding_length=self.padding_length,
        )


class SpectrogramDataset(BaseSpectrogramDataset):
    def __init__(
        self,
        inputs: Sequence[Union[InputData, LazyInputData]],
        sampling_length: int,
        min_not_silence_length: int,
        padding_length: int,
    ):
        super().__init__(
            sampling_length=sampling_length,
            min_not_silence_length=min_not_silence_length,
            padding_length=padding_length,
        )
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInputData):
            input = input.generate()

        return self.make_input(
            spectrogram_data=input.spectrogram, silence_data=input.silence,
        )


class TrainDataset(Dataset):
    def __init__(
        self,
        padded_spectrogram_dataset: BaseSpectrogramDataset,
        spectrogram_dataset: BaseSpectrogramDataset,
        latent_size: int,
    ):
        self.padded_spectrogram_dataset = padded_spectrogram_dataset
        self.spectrogram_dataset = spectrogram_dataset
        self.latent_size = latent_size

    def __len__(self):
        return len(self.spectrogram_dataset)

    def __getitem__(self, i):
        x = self.padded_spectrogram_dataset[i]["spectrogram"]
        x_ref1 = self.spectrogram_dataset[random.randrange(len(self))]["spectrogram"]
        x_ref2 = self.spectrogram_dataset[random.randrange(len(self))]["spectrogram"]
        z1 = generate_latent(self.latent_size)
        z2 = generate_latent(self.latent_size)
        return default_convert(dict(x=x, x_ref1=x_ref1, x_ref2=x_ref2, z1=z1, z2=z2))


def create_dataset(config: DatasetConfig, dataset_dir: Optional[Path]):
    spectrogram_paths = list(
        map(Path, sorted(config.spectrogram_filelist.read_text().split()))
    )
    assert len(spectrogram_paths) > 0

    silence_paths = list(map(Path, sorted(config.silence_filelist.read_text().split())))
    assert len(silence_paths) == len(spectrogram_paths)

    if dataset_dir is not None:
        spectrogram_paths = [Path.joinpath(dataset_dir, p) for p in spectrogram_paths]
        silence_paths = [Path.joinpath(dataset_dir, p) for p in silence_paths]

    assert tuple(p.stem for p in spectrogram_paths) == tuple(
        p.stem for p in silence_paths
    )

    inputs = [
        LazyInputData(spectrogram_path=spectrogram_path, silence_path=silence_path,)
        for spectrogram_path, silence_path in zip(spectrogram_paths, silence_paths)
    ]
    numpy.random.RandomState(config.seed).shuffle(inputs)

    num_test = config.num_test
    num_train = (
        config.num_train if config.num_train is not None else len(inputs) - num_test
    )

    trains = inputs[num_test:][:num_train]
    tests = inputs[:num_test]

    def make_dataset(data, for_evaluate=False):
        padded_spectrogram_dataset = SpectrogramDataset(
            inputs=data,
            sampling_length=config.sampling_length,
            min_not_silence_length=int(
                config.min_not_silence_rate * config.sampling_length
            ),
            padding_length=config.padding_length * 2,
        )

        spectrogram_dataset = SpectrogramDataset(
            inputs=data,
            sampling_length=config.sampling_length,
            min_not_silence_length=int(
                config.min_not_silence_rate * config.sampling_length
            ),
            padding_length=0,
        )
        dataset = TrainDataset(
            padded_spectrogram_dataset=padded_spectrogram_dataset,
            spectrogram_dataset=spectrogram_dataset,
            latent_size=config.latent_size,
        )

        if for_evaluate:
            dataset = ConcatDataset([dataset] * config.evaluate_times)

        return dataset

    return dict(
        train=make_dataset(trains),
        test=make_dataset(tests),
        eval=make_dataset(tests, for_evaluate=True),
    )
