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
This script computes audio tokens and stores them for TTS training.

$ python /NeMo/scripts/dataset_processing/tts/compute_audio_tokens.py \
    --manifest_path=train_manifest.json \
    --audio_dir=/data/audio \
    --feature_dir=/data/features \
    --feature_name="audio_tokens" \
    --model_path=/models/SpeechCodec.nemo \
    --volume_norm \
    --device=cuda:0 \
    --batch_size=16
"""

import argparse
import json
import string
import torch
from pathlib import Path

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest


remove_punct = str.maketrans('', '', string.punctuation)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute TTS features.",
    )
    parser.add_argument(
        "--input_manifest_path", required=True, type=Path, help="Path to input manifest.",
    )
    parser.add_argument(
        "--output_manifest_path", required=True, type=Path, help="Path to output manifest with transcribed text.",
    )
    parser.add_argument(
        "--audio_dir", required=True, type=Path, help="Path to base directory with audio data.",
    )
    parser.add_argument(
        "--model_path", type=Path, help="Path to checkpoint to load.",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Device to run model on.",
    )
    parser.add_argument(
        "--batch_size", required=True, type=int, help="Batch size to use during inference.",
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, help="Whether to overwrite existing manifest",
    )
    args = parser.parse_args()
    return args


def get_entries_sorted_by_duration(manifest_path):
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as input_f:
        for line in input_f:
            entry = json.loads(line)
            entries.append(entry)
    entries.sort(key=lambda entry: entry["duration"], reverse=True)
    return entries


def main():
    args = get_args()
    input_manifest_path = args.input_manifest_path
    output_manifest_path = args.output_manifest_path
    audio_dir = args.audio_dir
    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    overwrite = args.overwrite

    if not input_manifest_path.exists():
        raise ValueError(f"Manifest {input_manifest_path} does not exist.")

    if not audio_dir.exists():
        raise ValueError(f"Audio directory {audio_dir} does not exist.")

    if output_manifest_path.exists() and not overwrite:
        raise ValueError(f"Output manifest {output_manifest_path} exists.")

    asr_model = ASRModel.restore_from(model_path, map_location=device).eval()

    print(f"Reading manifest file {input_manifest_path}")
    #entries = read_manifest(input_manifest_path)
    entries = get_entries_sorted_by_duration(input_manifest_path)
    entries = entries[:1000]

    filepath_list = [str(audio_dir / entry["audio_filepath"]) for entry in entries]

    with torch.no_grad():
        transcripts, _ = asr_model.transcribe(
            paths2audio_files=filepath_list, batch_size=batch_size, return_hypotheses=False,
        )

    for entry, asr_text in zip(entries, transcripts):
        text = entry["normalized_text"]
        entry["asr_text"] = asr_text
        text = text.lower().strip().translate(remove_punct)
        asr_text = asr_text.lower().strip().translate(remove_punct)
        wer = word_error_rate(hypotheses=[asr_text], references=[text])
        cer = word_error_rate(hypotheses=[asr_text], references=[text], use_cer=True)
        entry["wer"] = round(wer, 3)
        entry["cer"] = round(cer, 3)

    entries.sort(key=lambda row: (row["audio_filepath"]))
    write_manifest(output_path=output_manifest_path, target_manifest=entries, ensure_ascii=False)


if __name__ == "__main__":
    main()
