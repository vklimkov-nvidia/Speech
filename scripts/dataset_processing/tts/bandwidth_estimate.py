# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is used to preprocess text before TTS model training. This is needed mainly for text normalization,
which is slow to rerun during training.

The output manifest will be the same as the input manifest but with final text stored in the 'normalized_text' field.

$ python <nemo_root_path>/scripts/dataset_processing/tts/preprocess_text.py \
    --input_manifest="<data_root_path>/manifest.json" \
    --output_manifest="<data_root_path>/manifest_processed.json" \
    --normalizer_config_path="<nemo_root_path>/examples/tts/conf/text/normalizer_en.yaml" \
    --lower_case \
    --num_workers=4 \
    --joblib_batch_size=16
"""

import argparse
import librosa
import numpy as np
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        "The package `nemo_text_processing` was not installed in this environment. Please refer to"
        " https://github.com/NVIDIA/NeMo-text-processing and install this package before using "
        "this script"
    )

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Process and normalize text data.",
    )
    parser.add_argument(
        "--input_manifest", required=True, type=Path, help="Path to input training manifest.",
    )
    parser.add_argument(
        "--output_manifest", required=True, type=Path, help="Path to output training manifest with estimated bandwidth.",
    )
    parser.add_argument("--input_audio_dir", required=True, type=Path, help="Path to directory containing audio.",)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="Whether to overwrite the output manifest file if it exists.",
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Number of parallel threads to use. If -1 all CPUs are used."
    )
    parser.add_argument(
        "--joblib_batch_size", type=int, help="Batch size for joblib workers. Defaults to 'auto' if not provided."
    )
    parser.add_argument(
        "--max_entries", default=0, type=int, help="If provided, maximum number of entries in the manifest to process."
    )

    args = parser.parse_args()
    return args


def estimate_bandwidth(audio_path, sample_rate=44100, n_fft=512, hop_length=441, top_db=100.0, frequency_threshold=-50.0):
    audio, _ = librosa.load(path=audio_path, sr=sample_rate)
    spec = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window="blackmanharris")
    power_spec = np.abs(spec) ** 2
    power_spec = np.mean(power_spec, axis=1)
    power_spec = librosa.power_to_db(power_spec, ref=n_fft, top_db=top_db)

    bandwidth = 0
    peak = np.max(power_spec)
    freq_width = sample_rate / n_fft
    for idx in range(len(power_spec) - 1, -1, -1):
        if power_spec[idx] - peak > frequency_threshold:
            bandwidth = idx * freq_width
            break

    return bandwidth


def _process_entry(entry: dict, audio_dir) -> dict:
    audio_filepath = entry["audio_filepath"]
    audio_path = audio_dir / audio_filepath
    bandwidth = estimate_bandwidth(audio_path=audio_path)
    entry["bandwidth"] = int(bandwidth)
    return entry


def main():
    args = get_args()

    input_manifest_path = args.input_manifest
    output_manifest_path = args.output_manifest
    input_audio_dir = args.input_audio_dir
    num_workers = args.num_workers
    batch_size = args.joblib_batch_size
    max_entries = args.max_entries
    overwrite = args.overwrite

    if output_manifest_path.exists():
        if overwrite:
            print(f"Will overwrite existing manifest path: {output_manifest_path}")
        else:
            raise ValueError(f"Manifest path already exists: {output_manifest_path}")

    entries = read_manifest(input_manifest_path)
    if max_entries:
        entries = entries[:max_entries]

    if not batch_size:
        batch_size = 'auto'

    output_entries = Parallel(n_jobs=num_workers, batch_size=batch_size)(
        delayed(_process_entry)(
            entry=entry,
            audio_dir=input_audio_dir,
        )
        for entry in tqdm(entries)
    )

    write_manifest(output_path=output_manifest_path, target_manifest=output_entries, ensure_ascii=False)


if __name__ == "__main__":
    main()
