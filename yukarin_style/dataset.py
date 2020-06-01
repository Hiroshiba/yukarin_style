from dataclasses import dataclass
import glob
from pathlib import Path
import random
from typing import Sequence, Union

from acoustic_feature_extractor.data.sampling_data import SamplingData
import numpy
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
            spectrogram=SamplingData.load(self.spectrogram_path),
            silence=SamplingData.load(self.silence_path),
        )


def extract_input(
    sampling_length: int,
    spectrogram_data: SamplingData,
    silence_data: SamplingData,
    min_not_silence_length: int,
):
    """
    :return:
        spectrogram: (sampling_length, ?)
        silence: (sampling_length, )
    """
    assert spectrogram_data.rate == silence_data.rate

    length = len(spectrogram_data.array)
    assert length >= sampling_length

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

    spectrogram = spectrogram_data.array[offset : offset + sampling_length]
    return dict(spectrogram=spectrogram, silence=silence)


def generate_latent(latent_size: int):
    return numpy.random.randn(latent_size).astype(numpy.float32)


class BaseSpectrogramDataset(Dataset):
    def __init__(
        self, sampling_length: int, min_not_silence_length: int,
    ):
        self.sampling_length = sampling_length
        self.min_not_silence_length = min_not_silence_length

    def make_input(
        self, spectrogram_data: SamplingData, silence_data: SamplingData,
    ):
        return extract_input(
            sampling_length=self.sampling_length,
            spectrogram_data=spectrogram_data,
            silence_data=silence_data,
            min_not_silence_length=self.min_not_silence_length,
        )


class SpectrogramDataset(BaseSpectrogramDataset):
    def __init__(
        self,
        inputs: Sequence[Union[InputData, LazyInputData]],
        sampling_length: int,
        min_not_silence_length: int,
    ):
        super().__init__(
            sampling_length=sampling_length,
            min_not_silence_length=min_not_silence_length,
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


def create_dataset(config: DatasetConfig):
    spectrogram_paths = {
        Path(p).stem: Path(p) for p in glob.glob(str(config.spectrogram_glob))
    }
    fn_list = sorted(spectrogram_paths.keys())
    assert len(fn_list) > 0

    silence_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.silence_glob))}
    assert set(fn_list) == set(silence_paths.keys())

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    num_test = config.num_test
    num_train = (
        config.num_train if config.num_train is not None else len(fn_list) - num_test
    )

    trains = fn_list[num_test:][:num_train]
    tests = fn_list[:num_test]

    def make_dataset(fns, for_evaluate=False):
        inputs = [
            LazyInputData(
                spectrogram_path=spectrogram_paths[fn], silence_path=silence_paths[fn],
            )
            for fn in fns
        ]

        sampling_length = config.sampling_length + config.padding_length * 4
        padded_spectrogram_dataset = SpectrogramDataset(
            inputs=inputs,
            sampling_length=sampling_length,
            min_not_silence_length=int(config.min_not_silence_rate * sampling_length),
        )

        sampling_length = config.sampling_length
        spectrogram_dataset = SpectrogramDataset(
            inputs=inputs,
            sampling_length=sampling_length,
            min_not_silence_length=int(config.min_not_silence_rate * sampling_length),
        )
        dataset = TrainDataset(
            padded_spectrogram_dataset=padded_spectrogram_dataset,
            spectrogram_dataset=spectrogram_dataset,
            latent_size=config.latent_size,
        )

        if for_evaluate:
            dataset = ConcatDataset([dataset] * config.evaluate_times)

        return dataset

    return dict(train=make_dataset(trains), test=make_dataset(tests))
