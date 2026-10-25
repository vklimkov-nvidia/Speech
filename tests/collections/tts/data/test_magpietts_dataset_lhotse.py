# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import random
from io import BytesIO

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from lhotse import CutSet, MonoCut, SupervisionSegment
from lhotse.array import Array, TemporalArray
from lhotse.testing.dummies import dummy_cut, dummy_recording
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.dataloader import resample
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import MagpieTTSLhotseDataset
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset


pytestmark = pytest.mark.unit

SAMPLE_RATE = 24000
CODEC_MODEL_SAMPLES_PER_FRAME = 480
CODEC_MODEL_INPUT_SAMPLE_RATE = 24000
FRAME_STACKING_FACTOR = 1
NUM_AUDIO_CODEBOOKS = 8

BPE_TOKENIZER_NAME = "nemotron_bpe"
BPE_TOKENIZER_MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
try:
    # Attempt to resolve local cache for CI tests
    BPE_TOKENIZER_MODEL = snapshot_download(BPE_TOKENIZER_MODEL, local_files_only=True)
except LocalEntryNotFoundError:
    # For local tests, can call into HF servers to download as needed
    pass


class _FakeIPATokenizer:
    pad = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text):
        return [7, 8] if text else []


class _FakeTextTokenizer:
    tokens = list(range(100))
    pad = 0
    tokenizer_pad_ids = {BPE_TOKENIZER_NAME: 0}

    def encode(self, text, tokenizer_name):
        return [10 + len(text)]


class _ManyTokenTextTokenizer(_FakeTextTokenizer):
    def encode(self, text, tokenizer_name):
        return list(range(10, 30))


def _seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def _tokenizer_config():
    return OmegaConf.create(
        {
            BPE_TOKENIZER_NAME: {
                "_target_": "AutoTokenizer",
                "pretrained_model": BPE_TOKENIZER_MODEL,
            }
        }
    )


def _memory_temporal_array(values, frame_shift=0.02):
    buffer = BytesIO()
    np.save(buffer, values)
    return TemporalArray(
        array=Array(
            storage_type="memory_npy",
            storage_path="",
            storage_key=buffer.getvalue(),
            shape=list(values.shape),
        ),
        temporal_dim=-1,
        frame_shift=frame_shift,
        start=0,
    )


def _cached_codes(num_codebooks=NUM_AUDIO_CODEBOOKS, num_frames=5, offset=0):
    codes = np.arange(num_codebooks * num_frames, dtype=np.int32).reshape(num_codebooks, num_frames)
    return (codes + offset) % 16


def _single_turn_cutset():
    cut = dummy_cut(
        0,
        duration=0.5,
        recording=dummy_recording(0, duration=0.5, with_data=True, sampling_rate=SAMPLE_RATE),
    )
    cut.target_audio = dummy_recording(10, duration=0.5, with_data=True, sampling_rate=SAMPLE_RATE)
    cut.supervisions = [
        SupervisionSegment(
            id="single-turn",
            recording_id=cut.recording_id,
            start=0.0,
            duration=0.2,
            text="hello",
            language="en",
            speaker="| Language:en Dataset:Unit Speaker:spk |",
            custom={"context_text": "speaker prompt", "normalized_text": "hello", "ipa": "həloʊ"},
        )
    ]
    cut.custom = {
        **(cut.custom or {}),
        "target_codes": _memory_temporal_array(_cached_codes(num_frames=4)),
        "context_codes": _memory_temporal_array(_cached_codes(num_frames=3, offset=3)),
        "tokenizer_names": [BPE_TOKENIZER_NAME],
        "lang": "en",
    }
    return CutSet.from_cuts([cut])


def _text_only_cutset(include_ipa=True):
    supervision_custom = {
        "normalized_text_reference": "the twenty fourth item",
        "text_only": True,
    }
    if include_ipa:
        supervision_custom["ipa"] = "ðə twɛnti fɔɹθ aɪtəm"
    cut = MonoCut(
        id="text-only-cut",
        start=0.0,
        duration=1.0,
        channel=0,
        recording=None,
        supervisions=[
            SupervisionSegment(
                id="text-only-sup",
                recording_id="text-only-cut",
                start=0.0,
                duration=1.0,
                text="the 24th item",
                language="en",
                speaker="| Language:en Dataset:SyntheticTN Speaker:synthetic |",
                custom=supervision_custom,
            )
        ],
        custom={"text_only": True},
    )
    return CutSet.from_cuts([cut])


def test_resample_preserves_text_only_cut_without_recording():
    cut = next(iter(_text_only_cutset()))

    resampled = resample(cut, sampling_rate=16000)

    assert resampled is cut
    assert not resampled.has_recording


def _multiturn_cutset():
    cut = dummy_cut(
        1,
        duration=0.8,
        recording=dummy_recording(1, duration=0.8, with_data=True, sampling_rate=SAMPLE_RATE),
    )
    cut.target_audio = dummy_recording(11, duration=0.8, with_data=True, sampling_rate=SAMPLE_RATE)
    cut.supervisions = [
        SupervisionSegment(
            id="turn-user-0",
            recording_id=cut.recording_id,
            start=0.0,
            duration=0.2,
            text="hi",
            language="en",
            speaker="user",
            custom={"context_text": "chat prompt"},
        ),
        SupervisionSegment(
            id="turn-agent-0",
            recording_id=cut.recording_id,
            start=0.3,
            duration=0.2,
            text="hello",
            language="en",
            speaker="assistant",
            custom={"ipa": "həloʊ", "ipa_alignment": [[0, 5, "hello", "həloʊ"]]},
        ),
        SupervisionSegment(
            id="turn-user-1",
            recording_id=cut.recording_id,
            start=0.55,
            duration=0.1,
            text="ok",
            language="en",
            speaker="user",
        ),
        SupervisionSegment(
            id="turn-agent-1",
            recording_id=cut.recording_id,
            start=0.68,
            duration=0.1,
            text="okay",
            language="en",
            speaker="assistant",
            custom={"reward": 0.5},
        ),
    ]
    cut.custom = {
        **(cut.custom or {}),
        "task": "dialog",
        "target_codes": _memory_temporal_array(_cached_codes(num_frames=8)),
        "source_codes": _memory_temporal_array(_cached_codes(num_frames=8, offset=1)),
        "context_codes": _memory_temporal_array(_cached_codes(num_frames=4, offset=2)),
        "tokenizer_names": [BPE_TOKENIZER_NAME],
        "lang": "en",
    }
    return CutSet.from_cuts([cut])


def _multiturn_cutset_without_context_codes():
    """Multiturn cut carrying `context_text` and `target_codes`, but no `context_codes`."""
    cut = dummy_cut(
        2,
        duration=0.8,
        recording=dummy_recording(2, duration=0.8, with_data=True, sampling_rate=SAMPLE_RATE),
    )
    cut.target_audio = dummy_recording(12, duration=0.8, with_data=True, sampling_rate=SAMPLE_RATE)
    cut.supervisions = [
        SupervisionSegment(
            id="turn-user-0",
            recording_id=cut.recording_id,
            start=0.0,
            duration=0.05,
            text="hi",
            language="en",
            speaker="user",
            custom={"context_text": "speaker prompt"},
        ),
        SupervisionSegment(
            id="turn-agent-0",
            recording_id=cut.recording_id,
            start=0.06,
            duration=0.1,
            text="hello",
            language="en",
            speaker="assistant",
        ),
    ]
    cut.custom = {
        **(cut.custom or {}),
        "task": "dialog",
        "target_codes": _memory_temporal_array(_cached_codes(num_frames=8)),
        "source_codes": _memory_temporal_array(_cached_codes(num_frames=8, offset=1)),
        "tokenizer_names": [BPE_TOKENIZER_NAME],
        "lang": "en",
    }
    return CutSet.from_cuts([cut])


def _dataset_kwargs():
    return {
        "sample_rate": SAMPLE_RATE,
        "volume_norm": False,
        "codec_model_samples_per_frame": CODEC_MODEL_SAMPLES_PER_FRAME,
        "num_audio_codebooks": NUM_AUDIO_CODEBOOKS,
        "prior_scaling_factor": None,
        "load_cached_codes_if_available": True,
        "dataset_type": "train",
        "load_16khz_audio": False,
        "pad_context_text_to_max_duration": False,
        "context_duration_min": 0.04,
        "context_duration_max": 0.04,
        "use_text_conditioning_tokenizer": True,
        "text_conditioning_tokenizer_name": BPE_TOKENIZER_NAME,
        "tokenizer_config": _tokenizer_config(),
    }


def _multiturn_dataset_kwargs():
    kwargs = _dataset_kwargs()
    kwargs.update(
        {
            "codec_model_input_sample_rate": CODEC_MODEL_INPUT_SAMPLE_RATE,
            "frame_stacking_factor": FRAME_STACKING_FACTOR,
            "source_sample_rate": SAMPLE_RATE,
            "input_roles": ["user"],
            "output_roles": ["assistant"],
            "add_text_bos": False,
        }
    )
    return kwargs


class TestMagpieTTSLhotseDatasets:
    def test_text_only_cut_uses_raw_text_ipa_and_dummy_audio_codes(self):
        _seed_everything()
        kwargs = _dataset_kwargs()
        kwargs.update(
            {
                "frame_stacking_factor": 2,
                "phoneme_stacking_factor": 2,
                "text_span_corruption_prob": 1.0,
            }
        )
        dataset = MagpieTTSLhotseDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.pad_id = dataset.text_tokenizer.pad

        batch = dataset[_text_only_cutset()]

        # The synthetic normalization pair must train raw written text -> IPA.
        assert batch["raw_texts"] == ["the 24th item"]
        assert batch["phoneme_tokens"][0, :4].tolist() == [1, 7, 8, 2]
        assert batch["text_only"].tolist() == [True]
        assert batch["text_span_corrupted"].tolist() == [False]
        assert not batch["text_span_corruption_mask"].any()
        # Four IPA tokens stack to two supervised phoneme steps, which
        # require two raw codec frames when frame_stacking_factor=2.
        assert batch["audio_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 2)
        assert batch["audio_codes_lens"].tolist() == [2]

    def test_text_only_cut_supports_on_the_fly_codec_extraction(self):
        _seed_everything()
        kwargs = _dataset_kwargs()
        kwargs.update(
            {
                "load_cached_codes_if_available": False,
                "frame_stacking_factor": 2,
                "phoneme_stacking_factor": 2,
            }
        )
        dataset = MagpieTTSLhotseDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.pad_id = dataset.text_tokenizer.pad

        batch = dataset[_text_only_cutset()]

        expected_samples = 2 * kwargs["codec_model_samples_per_frame"]
        assert "audio_codes" not in batch
        assert batch["audio"].shape == (1, expected_samples)
        assert batch["audio_lens"].tolist() == [expected_samples]
        assert batch["audio"].eq(0.0).all()
        assert "context_audio_codes" not in batch
        assert batch["context_audio"].shape == (1, kwargs["codec_model_samples_per_frame"])

    def test_text_only_cut_requires_ipa(self):
        dataset = MagpieTTSLhotseDataset(**_dataset_kwargs())
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.pad_id = dataset.text_tokenizer.pad

        with pytest.raises(ValueError, match="requires a non-empty supervision custom.ipa"):
            dataset[_text_only_cutset(include_ipa=False)]

    def test_multiturn_dataset_supports_recordingless_text_only_cut(self):
        _seed_everything()
        kwargs = _multiturn_dataset_kwargs()
        kwargs.update(
            {
                "output_roles": ["assistant", "agent"],
                "load_cached_codes_if_available": False,
                "text_span_corruption_prob": 1.0,
            }
        )
        dataset = MagpieTTSLhotseMultiturnDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.cfg_unk_token_id = dataset.bos_id + 2
        dataset.interruption_token_id = dataset.bos_id + 3
        dataset.pad_id = dataset.text_tokenizer.pad

        cuts = _text_only_cutset()
        next(iter(cuts)).context_audio = dummy_recording(20, duration=0.2, with_data=True, sampling_rate=SAMPLE_RATE)
        batch = dataset[cuts]

        assert batch["raw_texts"] == ["the 24th item"]
        assert batch["text_only"].tolist() == [True]
        assert batch["text_span_corrupted"].tolist() == [False]
        assert not batch["text_span_corruption_mask"].any()
        assert batch["audio_lens"].tolist() == [SAMPLE_RATE]
        assert batch["audio"].eq(0.0).all()
        assert batch["source_audio"].eq(0.0).all()
        assert batch["context_audio_lens"].tolist() == [int(0.04 * SAMPLE_RATE)]
        non_padding_phonemes = batch["phoneme_tokens"][0]
        non_padding_phonemes = non_padding_phonemes[non_padding_phonemes != dataset.phoneme_tokenizer.pad]
        assert non_padding_phonemes.tolist() == [1, 7, 8, 2]

    def test_multiturn_dataset_text_span_corruption_masks_audio_backed_tts_cut(self):
        _seed_everything()
        kwargs = _multiturn_dataset_kwargs()
        kwargs.update(
            {
                "output_roles": ["assistant", "agent"],
                "text_span_corruption_prob": 1.0,
                "text_span_corruption_short_max_tokens": 3,
                "text_span_corruption_long_max_tokens": 8,
                "text_span_corruption_long_prob": 0.0,
            }
        )
        dataset = MagpieTTSLhotseMultiturnDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.cfg_unk_token_id = dataset.bos_id + 2
        dataset.interruption_token_id = dataset.bos_id + 3
        dataset.pad_id = dataset.text_tokenizer.pad

        batch = dataset[_single_turn_cutset()]
        corruption_mask = batch["text_span_corruption_mask"][0]

        assert batch["text_only"].tolist() == [False]
        assert batch["text_span_corrupted"].tolist() == [True]
        assert corruption_mask.sum().item() == 1
        assert not corruption_mask[batch["text"][0].eq(dataset.eos_id)].any()

    def test_text_span_corruption_masks_short_non_eos_span_when_ipa_is_available(self):
        _seed_everything()
        kwargs = _dataset_kwargs()
        kwargs.update(
            {
                "text_span_corruption_prob": 1.0,
                "text_span_corruption_short_max_tokens": 3,
                "text_span_corruption_long_max_tokens": 8,
                "text_span_corruption_long_prob": 0.0,
            }
        )
        dataset = MagpieTTSLhotseDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.pad_id = dataset.text_tokenizer.pad

        batch = dataset[_single_turn_cutset()]
        corruption_mask = batch["text_span_corruption_mask"][0]

        assert batch["text_span_corrupted"].tolist() == [True]
        assert corruption_mask.sum().item() == 1
        assert not corruption_mask[batch["text_lens"][0] - 1]

    def test_text_span_corruption_can_select_longer_span(self):
        _seed_everything()
        kwargs = _dataset_kwargs()
        kwargs.update(
            {
                "text_span_corruption_prob": 1.0,
                "text_span_corruption_short_max_tokens": 3,
                "text_span_corruption_long_max_tokens": 8,
                "text_span_corruption_long_prob": 1.0,
            }
        )
        dataset = MagpieTTSLhotseDataset(**kwargs)
        dataset.text_tokenizer = _ManyTokenTextTokenizer()
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.pad_id = dataset.text_tokenizer.pad

        batch = dataset[_single_turn_cutset()]
        corruption_mask = batch["text_span_corruption_mask"][0]
        masked_positions = corruption_mask.nonzero(as_tuple=False).squeeze(1)

        assert 4 <= masked_positions.numel() <= 8
        assert torch.equal(masked_positions, torch.arange(masked_positions[0], masked_positions[-1] + 1))
        assert not corruption_mask[batch["text_lens"][0] - 1]

    def test_single_turn_dataset_uses_bpe_and_cached_codes(self):
        _seed_everything()
        dataset = MagpieTTSLhotseDataset(**_dataset_kwargs())

        batch = dataset[_single_turn_cutset()]

        assert batch["dataset_names"] == ["Unit"]
        assert batch["languages"] == ["en"]
        assert batch["raw_texts"] == ["hello"]
        assert "audio" not in batch
        assert "context_audio" not in batch
        assert batch["audio_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 4)
        assert batch["audio_codes_lens"].tolist() == [4]
        assert batch["context_audio_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 2)
        assert batch["context_audio_codes_lens"].tolist() == [2]
        assert batch["text"].shape[0] == 1
        assert batch["text_lens"].item() > 0
        assert batch["context_text_tokens"].shape[0] == 1
        assert batch["context_text_tokens_lens"].item() > 0
        assert batch["has_text_context"].tolist() == [True]

    @pytest.mark.parametrize(
        "add_language_to_context_text, expected_context_text",
        [(True, "[EN]"), (False, "[NO TEXT CONTEXT]")],
    )
    def test_single_turn_dataset_can_ignore_manifest_context_text(
        self, add_language_to_context_text, expected_context_text
    ):
        _seed_everything()
        kwargs = _dataset_kwargs()
        kwargs.update(
            {
                "add_language_to_context_text": add_language_to_context_text,
                "ignore_manifest_context_text": True,
            }
        )
        dataset = MagpieTTSLhotseDataset(**kwargs)

        batch = dataset[_single_turn_cutset()]

        expected_tokens = dataset.text_tokenizer.encode(expected_context_text, tokenizer_name=BPE_TOKENIZER_NAME)
        assert batch["has_text_context"].tolist() == [False]
        assert batch["context_text_tokens"][0].tolist() == expected_tokens
        # Audio conditioning must be untouched by the context text override.
        assert batch["context_audio_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 2)

    def test_single_turn_dataset_uses_manifest_context_text_by_default(self):
        _seed_everything()
        dataset = MagpieTTSLhotseDataset(**_dataset_kwargs())

        batch = dataset[_single_turn_cutset()]

        expected_tokens = dataset.text_tokenizer.encode("speaker prompt", tokenizer_name=BPE_TOKENIZER_NAME)
        assert batch["has_text_context"].tolist() == [True]
        assert batch["context_text_tokens"][0].tolist() == expected_tokens

    def test_multiturn_ignoring_context_text_enables_audio_context_fallback(self):
        _seed_everything()
        kwargs = _multiturn_dataset_kwargs()
        cutset = _multiturn_cutset_without_context_codes()

        # A cut with `context_text` is treated as a text-context sample, so no context audio is derived.
        text_context_batch = MagpieTTSLhotseMultiturnDataset(**kwargs)[cutset]
        assert text_context_batch["has_text_context"].tolist() == [True]
        assert text_context_batch["context_audio_codes_lens"].tolist() == [0]

        kwargs["ignore_manifest_context_text"] = True
        audio_context_batch = MagpieTTSLhotseMultiturnDataset(**kwargs)[cutset]
        assert audio_context_batch["has_text_context"].tolist() == [False]
        assert audio_context_batch["context_audio_codes_lens"].tolist() == [5]

    def test_multiturn_pronunciation_control_only_changes_target_turns(self):
        _seed_everything()
        kwargs = _dataset_kwargs()
        kwargs.update(
            {
                "codec_model_input_sample_rate": CODEC_MODEL_INPUT_SAMPLE_RATE,
                "frame_stacking_factor": FRAME_STACKING_FACTOR,
                "source_sample_rate": SAMPLE_RATE,
                "input_roles": ["user"],
                "output_roles": ["assistant"],
                "add_text_bos": False,
                "use_text_conditioning_tokenizer": False,
                "enable_phoneme_text_input": True,
                "partial_phoneme_text_prob": 1.0,
                "partial_phoneme_portion_min": 1.0,
                "partial_phoneme_portion_max": 1.0,
            }
        )
        dataset = MagpieTTSLhotseMultiturnDataset(**kwargs)
        dataset.text_tokenizer = _FakeTextTokenizer()
        dataset.phoneme_tokenizer = _FakeIPATokenizer()
        dataset.bos_id = len(dataset.text_tokenizer.tokens)
        dataset.eos_id = dataset.bos_id + 1
        dataset.cfg_unk_token_id = dataset.bos_id + 2
        dataset.interruption_token_id = dataset.bos_id + 3
        dataset.pad_id = dataset.text_tokenizer.pad
        dataset.text_phoneme_token_offset = dataset.bos_id + 4

        batch = dataset[_multiturn_cutset()]

        target_tokens = batch["text"][0]
        source_tokens = batch["source_tokens"][0]
        assert target_tokens[15:17].tolist() == [
            dataset.text_phoneme_token_offset + 7,
            dataset.text_phoneme_token_offset + 8,
        ]
        assert torch.all(source_tokens < dataset.text_phoneme_token_offset)
        assert target_tokens[25].item() == dataset.interruption_token_id

    def test_multiturn_dataset_uses_bpe_and_cached_codes(self):
        _seed_everything()
        kwargs = _dataset_kwargs()
        kwargs.update(
            {
                "codec_model_input_sample_rate": CODEC_MODEL_INPUT_SAMPLE_RATE,
                "frame_stacking_factor": FRAME_STACKING_FACTOR,
                "source_sample_rate": SAMPLE_RATE,
                "input_roles": ["user"],
                "output_roles": ["assistant"],
                "add_text_bos": False,
            }
        )
        dataset = MagpieTTSLhotseMultiturnDataset(**kwargs)

        batch = dataset[_multiturn_cutset()]

        assert batch["sample_id"] == ["dummy-mono-cut-0001"]
        assert batch["dataset_names"] == ["unknown"]
        assert batch["languages"] == ["en"]
        assert batch["raw_texts"] == ["hello okay"]
        assert batch["task"] == ["dialog"]
        assert batch["audio"].shape[0] == 1
        assert batch["source_audio"].shape[0] == 1
        assert batch["audio_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 8)
        assert batch["audio_codes_lens"].tolist() == [8]
        assert batch["source_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 8)
        assert batch["source_codes_lens"].tolist() == [8]
        assert batch["context_audio_codes"].shape == (1, NUM_AUDIO_CODEBOOKS, 2)
        assert batch["context_audio_codes_lens"].tolist() == [2]
        assert batch["source_tokens"].shape[0] == 1
        assert batch["source_token_lens"].item() > 0
        assert batch["text"].shape[0] == 1
        assert batch["text_lens"].item() > 0
        assert batch["agent_mask"].shape == batch["user_mask"].shape
        assert batch["agent_mask_lens"].tolist() == batch["user_mask_lens"].tolist()
        assert batch["agent_mask"].sum().item() > 0
        assert batch["user_mask"].sum().item() > 0
        assert batch["user_audio_turn_splitted"].shape[0] == 2
        assert batch["user_audio_turn_splitted_lens"].tolist() == [4800, 2400]
        assert batch["user_audio_turn_splitted_indices"].shape == (2, 3)
        assert batch["context_text_tokens"].shape[0] == 1
        assert batch["context_text_tokens_lens"].item() > 0
        assert batch["has_text_context"].tolist() == [True]
        torch.testing.assert_close(batch["rewards"], torch.tensor([0.5], device=batch["rewards"].device))
