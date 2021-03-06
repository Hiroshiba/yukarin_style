import argparse
from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy
from acoustic_feature_extractor.data.sampling_data import SamplingData
from tqdm import tqdm


def process(args: Tuple[int, Path], sampling_lengths: Sequence[int]):
    i_data, path = args
    vector = numpy.empty(len(sampling_lengths), dtype=numpy.int32)

    data = SamplingData.load(path)
    array = ~numpy.squeeze(data.array)
    for i_length, sampling_length in enumerate(sampling_lengths):
        m = numpy.convolve(
            numpy.ones(sampling_length, dtype=numpy.int32), array, mode="valid"
        ).max()
        vector[i_length] = m

    return i_data, vector


def check_min_silence_length(
    input_glob: str, sampling_lengths: List[int],
):
    print("start check_min_silence_length.py")

    num_lengths = len(sampling_lengths)

    paths = list(map(Path, glob(input_glob)))

    num_data = len(paths)
    matrix = numpy.empty((num_data, num_lengths), dtype=numpy.int32)
    matrix.fill(numpy.nan)

    wrapper = partial(process, sampling_lengths=sampling_lengths)

    with Pool() as pool:
        it = pool.imap_unordered(wrapper, enumerate(paths))
        it = tqdm(it, total=len(paths))

        for i_data, vector in it:
            matrix[i_data] = vector

    mins = numpy.min(matrix, axis=0)
    argmins = numpy.argmin(matrix, axis=0)
    for sampling_length, m, index in zip(sampling_lengths, mins, argmins):
        print(f"sampling_length: {sampling_length}, min: {m}, path: {paths[index]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--sampling_lengths", nargs="+", type=int, default=[64])
    check_min_silence_length(**vars(parser.parse_args()))
