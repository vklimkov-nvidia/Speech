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
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from pathlib import Path
import re
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer


LEADING_PUNCT_REG = r'^[^a-zA-Z]+ '
TRAILING_PUNCT_REG = r' [^a-zA-Z]+$'


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Process and normalize text data.",
    )
    parser.add_argument(
        "--input_manifest", required=True, type=Path, help="Path to input training manifest.",
    )
    parser.add_argument(
        "--output_manifest", required=True, type=Path, help="Path to output training manifest with processed text.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="Whether to overwrite the output manifest file if it exists.",
    )
    parser.add_argument(
        "--model_config_path", type=Path, help="Path to DiscreteSpeech config file.",
    )
    parser.add_argument(
        "--phoneme_dict_path", type=Path, help="Path to DiscreteSpeech config file.",
    )
    parser.add_argument(
        "--text_key", default="text", type=str, help="Input text field to process.",
    )
    parser.add_argument(
        "--min_duration", default=0.0, type=float, help="",
    )
    parser.add_argument(
        "--max_duration", default=30.0, type=float, help="",
    )
    parser.add_argument(
        "--min_words", default=2, type=str, help="Minimum number of words in text field.",
    )
    parser.add_argument(
        "--min_text_per_second", default=10, type=int, help="",
    )
    parser.add_argument(
        "--max_text_per_second", default=21, type=int, help="",
    )
    parser.add_argument(
        "--sanitize_text", action=argparse.BooleanOptionalAction, help="",
    )
    parser.add_argument(
        "--remove_oov", action=argparse.BooleanOptionalAction, help="",
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


def _sanitize_text(text: str):
    sanitized_text = re.sub(LEADING_PUNCT_REG, '', text)
    sanitized_text = re.sub(TRAILING_PUNCT_REG, '', sanitized_text)
    sanitized_text = sanitized_text.strip()
    return sanitized_text


def _process_entry(
    entry: dict,
    tokenizer: BaseTokenizer,
    text_key: str,
    min_duration: float,
    max_duration: float,
    min_words: int,
    min_text_per_second: int,
    max_text_per_second: int,
    sanitize_text: bool,
    remove_oov: bool,
):
    text = entry[text_key]

    duration = entry["duration"]
    if not min_duration <= duration <= max_duration:
        #print(f'Skipping duration: {duration}')
        return None

    num_words = len(text.split(" "))
    if num_words < min_words:
        #print(f'Text too short: {text}')
        return None

    tokens = tokenizer.encode(text)

    if remove_oov and tokenizer.oov in tokens:
        #print(f"Removing text '{text}'")
        return None

    num_tokens = len(tokens)
    min_text_len = int(min_text_per_second * duration)
    max_text_len = int(max_text_per_second * duration)
    if not (min_text_len <= num_tokens <= max_text_len):
        #print(f"Removing token length: {duration:.2f}, {num_tokens}, '{text}'")
        return None

    if sanitize_text:
        text = _sanitize_text(text)
        entry[text_key] = text

    return entry


def main():
    args = get_args()

    input_manifest_path = args.input_manifest
    output_manifest_path = args.output_manifest
    model_config_path = args.model_config_path
    phoneme_dict_path = args.phoneme_dict_path
    text_key = args.text_key
    min_duration = args.min_duration
    max_duration = args.max_duration
    min_words = args.min_words
    min_text_per_second = args.min_text_per_second
    max_text_per_second = args.max_text_per_second
    sanitize_text = args.sanitize_text
    remove_oov = args.remove_oov
    num_workers = args.num_workers
    batch_size = args.joblib_batch_size
    max_entries = args.max_entries
    overwrite = args.overwrite

    if output_manifest_path.exists():
        if overwrite:
            print(f"Will overwrite existing manifest path: {output_manifest_path}")
        else:
            raise ValueError(f"Manifest path already exists: {output_manifest_path}")

    model_config = OmegaConf.load(model_config_path)
    tokenizer_config = model_config.model.text_tokenizer
    tokenizer_config.g2p.phoneme_probability = 1.0
    tokenizer_config.g2p.phoneme_dict = phoneme_dict_path
    if "heteronyms" in tokenizer_config.g2p:
        del tokenizer_config.g2p.heteronyms
    print(tokenizer_config)
    tokenizer = instantiate(tokenizer_config)

    entries = read_manifest(input_manifest_path)
    if max_entries:
        entries = entries[:max_entries]

    if not batch_size:
        batch_size = 'auto'

    output_entries = Parallel(n_jobs=num_workers, batch_size=batch_size, backend="threading")(
        delayed(_process_entry)(
            entry=entry,
            tokenizer=tokenizer,
            text_key=text_key,
            min_duration=min_duration,
            max_duration=max_duration,
            min_words=min_words,
            min_text_per_second=min_text_per_second,
            max_text_per_second=max_text_per_second,
            sanitize_text=sanitize_text,
            remove_oov=remove_oov,
        )
        for entry in tqdm(entries)
    )
    output_entries = [entry for entry in output_entries if entry is not None]

    print(f"Original entries: {len(entries)}")
    print(f"Output entries: {len(output_entries)}")

    write_manifest(output_path=output_manifest_path, target_manifest=output_entries, ensure_ascii=False)


if __name__ == "__main__":
    main()
