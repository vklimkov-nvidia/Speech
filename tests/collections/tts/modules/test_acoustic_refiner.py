# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.magpietts_modules import AcousticRefiner, AcousticRefinerOrder

pytestmark = pytest.mark.unit

NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 32
NUM_SPECIAL_TOKENS = 4
D_MODEL = 16
MASK_TOKEN_ID = CODEBOOK_SIZE + 3
AUDIO_EOS_ID = CODEBOOK_SIZE + 1
# leaves one codebook for the backbone to predict
SCHEDULE = (3, 4)


class _CodeEmbedder(nn.Module):
    """Stands in for the backbone's ``embed_codes`` callable: (B, C, T) codes -> (B, T, d_model)."""

    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(CODEBOOK_SIZE + NUM_SPECIAL_TOKENS, d_model) for _ in range(NUM_CODEBOOKS)]
        )

    def forward(self, codes):
        embedded = self.embeddings[0](codes[:, 0, :])
        for codebook in range(1, codes.size(1)):
            embedded = embedded + self.embeddings[codebook](codes[:, codebook, :])
        return embedded / codes.size(1)


def _make_refiner(scale=1.0, **overrides):
    torch.manual_seed(0)
    kwargs = dict(
        n_layers=2,
        d_model=D_MODEL,
        d_ffn=4 * D_MODEL,
        sa_n_heads=4,
        embed_codes=_CodeEmbedder(),
        backbone_dim=D_MODEL,
        num_audio_codebooks=NUM_CODEBOOKS,
        audio_eos_id=AUDIO_EOS_ID,
        mask_token_id=MASK_TOKEN_ID,
        codebook_size=CODEBOOK_SIZE,
        prediction_schedule=SCHEDULE,
    )
    kwargs.update(overrides)
    refiner = AcousticRefiner(**kwargs)
    if scale != 1.0:
        # an untrained stack barely reacts to its context, which would hide context bugs
        with torch.no_grad():
            for parameter in refiner.transformers.parameters():
                parameter.mul_(scale)
    return refiner


def _cache(refiner, batch_size):
    return refiner.make_cache(batch_size, device=torch.device("cpu"), dtype=torch.float32)


def _codes(batch_size, frames):
    torch.manual_seed(1)
    return torch.randint(0, CODEBOOK_SIZE, (batch_size, frames, NUM_CODEBOOKS))


def test_rejects_dimension_mismatch_with_backbone():
    # the refiner adds no projection, so a mismatch has to be reported rather than papered over
    with pytest.raises(ValueError, match="has to match the backbone dimension"):
        _make_refiner(backbone_dim=D_MODEL * 2)


def test_rejects_schedule_that_leaves_nothing_for_the_backbone():
    with pytest.raises(ValueError, match="leaving none for the backbone"):
        _make_refiner(prediction_schedule=(4, 4))


@pytest.mark.parametrize("commit_order", [AcousticRefinerOrder.CODEBOOK, AcousticRefinerOrder.CONFIDENCE])
def test_compute_loss_is_finite_and_differentiable(commit_order):
    refiner = _make_refiner(commit_order=commit_order)
    target_codes = _codes(batch_size=3, frames=6)
    target_codes[:, -1, :] = AUDIO_EOS_ID  # trailing special-token frame carries no valid target
    hidden_states = torch.randn(3, 6, D_MODEL, requires_grad=True)

    loss = refiner.compute_loss(hidden_states, target_codes, torch.tensor([6, 4, 2]))
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(hidden_states.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in refiner.out_proj.parameters())


@pytest.mark.parametrize("commit_order", [AcousticRefinerOrder.CODEBOOK, AcousticRefinerOrder.CONFIDENCE])
def test_code_masking_hides_committed_codes_only_while_training(commit_order):
    """Masking the codes is a training-time regularizer, so evaluation has to be left untouched."""
    # a share of 1.0 hides every frame, leaving every step to predict from the hidden states alone
    refiner = _make_refiner(commit_order=commit_order, mask_codes=True, mask_min=1.0, mask_max=1.0)
    plain = _make_refiner(commit_order=commit_order)
    target_codes = _codes(batch_size=2, frames=5)
    hidden_states = torch.randn(2, 5, D_MODEL)
    lengths = torch.tensor([5, 4])

    refiner.eval()
    plain.eval()
    torch.testing.assert_close(
        refiner.compute_loss(hidden_states, target_codes, lengths),
        plain.compute_loss(hidden_states, target_codes, lengths),
    )

    refiner.train()
    masked = refiner.compute_loss(hidden_states, target_codes, lengths)
    masked.backward()

    assert torch.isfinite(masked) and masked > 0
    assert not torch.isclose(masked, plain.compute_loss(hidden_states, target_codes, lengths))
    assert all(torch.isfinite(parameter.grad).all() for parameter in refiner.out_proj.parameters())


def test_compute_loss_ignores_frames_beyond_length():
    refiner = _make_refiner()
    target_codes = _codes(batch_size=2, frames=5)
    hidden_states = torch.randn(2, 5, D_MODEL)
    lengths = torch.tensor([5, 3])

    padded = refiner.compute_loss(hidden_states, target_codes.clone(), lengths)
    scrambled = target_codes.clone()
    scrambled[1, 3:, :] = (scrambled[1, 3:, :] + 7) % CODEBOOK_SIZE
    torch.testing.assert_close(padded, refiner.compute_loss(hidden_states, scrambled, lengths))


@pytest.mark.parametrize("commit_order", [AcousticRefinerOrder.CODEBOOK, AcousticRefinerOrder.CONFIDENCE])
def test_refine_codes_commits_every_codebook_and_keeps_backbone_codes(commit_order):
    refiner = _make_refiner(commit_order=commit_order)
    pred_codes = _codes(batch_size=4, frames=1)

    refined = refiner.refine_codes(torch.randn(4, 1, D_MODEL), pred_codes)

    assert refined.shape == pred_codes.shape
    assert (refined != MASK_TOKEN_ID).all(), "every codebook has to be committed by the last step"
    assert (refined < CODEBOOK_SIZE).all(), "out_proj only covers codec tokens"
    backbone = slice(None, refiner.num_backbone_codebooks)
    torch.testing.assert_close(refined[:, :, backbone], pred_codes[:, :, backbone])


def test_refine_codes_passes_special_tokens_through_from_the_backbone():
    # EOS detection downstream relies on the backbone's codebooks surviving refinement
    refiner = _make_refiner()
    pred_codes = _codes(batch_size=2, frames=1)
    pred_codes[:, :, 0] = AUDIO_EOS_ID

    refined = refiner.refine_codes(torch.randn(2, 1, D_MODEL), pred_codes, temperature=0.0)

    assert (refined[:, :, 0] == AUDIO_EOS_ID).all()


def test_cached_streaming_matches_whole_sequence_refinement():
    """One frame at a time with the cache has to equal refining the whole sequence at once."""
    frames = 10
    hidden_states = torch.randn(1, frames, D_MODEL)
    pred_codes = _codes(batch_size=1, frames=frames)

    whole = _make_refiner(scale=6.0).refine_codes(hidden_states, pred_codes, temperature=0.0)

    refiner = _make_refiner(scale=6.0)
    cache = _cache(refiner, batch_size=1)
    streamed = torch.cat(
        [
            refiner.refine_codes(
                hidden_states[:, frame : frame + 1], pred_codes[:, frame : frame + 1], cache=cache, temperature=0.0
            )
            for frame in range(frames)
        ],
        dim=1,
    )

    assert AcousticRefiner.cached_frames(cache) == frames
    torch.testing.assert_close(streamed, whole)


def test_a_fresh_cache_starts_empty():
    refiner = _make_refiner()
    cache = _cache(refiner, batch_size=1)
    assert AcousticRefiner.cached_frames(cache) == 0

    refiner.refine_codes(torch.randn(1, 1, D_MODEL), _codes(batch_size=1, frames=1), cache=cache)
    assert AcousticRefiner.cached_frames(cache) == 1
    # a second cache is independent, so one utterance cannot disturb another
    assert AcousticRefiner.cached_frames(_cache(refiner, batch_size=1)) == 0


def test_a_warm_cache_takes_several_frames_at_once():
    """Frames batched behind a warm cache have to match stepping through them one by one."""
    frames, warm = 6, 3
    hidden_states = torch.randn(1, frames, D_MODEL)
    pred_codes = _codes(batch_size=1, frames=frames)

    def refine_tail(one_at_a_time):
        refiner = _make_refiner(scale=6.0)
        cache = _cache(refiner, batch_size=1)
        refiner.refine_codes(hidden_states[:, :warm], pred_codes[:, :warm], cache=cache, temperature=0.0)
        if not one_at_a_time:
            return refiner.refine_codes(hidden_states[:, warm:], pred_codes[:, warm:], cache=cache, temperature=0.0)
        return torch.cat(
            [
                refiner.refine_codes(
                    hidden_states[:, frame : frame + 1], pred_codes[:, frame : frame + 1], cache=cache, temperature=0.0
                )
                for frame in range(warm, frames)
            ],
            dim=1,
        )

    torch.testing.assert_close(refine_tail(one_at_a_time=True), refine_tail(one_at_a_time=False))


def test_cfg_takes_doubled_hidden_states_and_single_stream_codes():
    refiner = _make_refiner()
    num_streams = 3
    pred_codes = _codes(batch_size=num_streams, frames=1)

    refined = refiner.refine_codes(
        hidden_states=torch.randn(2 * num_streams, 1, D_MODEL),
        pred_codes=pred_codes,
        use_cfg=True,
        cfg_scale=2.5,
    )

    assert refined.shape == pred_codes.shape
    assert (refined != MASK_TOKEN_ID).all()


def test_cfg_rejects_codes_for_both_streams():
    refiner = _make_refiner()
    with pytest.raises(ValueError, match="rows of predicted codes"):
        refiner.refine_codes(
            hidden_states=torch.randn(6, 1, D_MODEL),
            pred_codes=_codes(batch_size=6, frames=1),
            use_cfg=True,
        )


def test_advance_caches_the_positions_it_is_given():
    refiner = _make_refiner()
    cache = _cache(refiner, batch_size=2)

    refiner.advance(torch.randn(2, 5, D_MODEL), cache=cache)
    assert AcousticRefiner.cached_frames(cache) == 5, "the advanced positions have to occupy the cache"

    refiner.refine_codes(torch.randn(2, 1, D_MODEL), _codes(batch_size=2, frames=1), cache=cache)
    assert AcousticRefiner.cached_frames(cache) == 6

    # a warm cache is no obstacle, so the positions of a later prefill go in as one pass too
    refiner.advance(torch.randn(2, 3, D_MODEL), cache=cache)
    assert AcousticRefiner.cached_frames(cache) == 9


def test_a_row_left_out_of_refinement_caches_what_advance_would():
    """A position without audio has to look the same whether it came from advance or a skipped row."""
    hidden_states = torch.randn(1, 3, D_MODEL)
    frame_codes = _codes(batch_size=1, frames=1)
    refine_frame = torch.tensor([True])

    def refine_last(advanced_frames, skipped):
        refiner = _make_refiner(scale=6.0)
        cache = _cache(refiner, batch_size=1)
        refiner.advance(hidden_states[:, :advanced_frames], cache=cache)
        for frame in range(advanced_frames, 2):
            skipped_codes = _codes(batch_size=1, frames=1) if skipped else frame_codes
            left_alone = refiner.refine_codes(
                hidden_states[:, frame : frame + 1],
                skipped_codes,
                cache=cache,
                temperature=0.0,
                refine=torch.tensor([False]),
            )
            assert (left_alone == MASK_TOKEN_ID).all(), "a row left alone keeps the mask token"
        return refiner.refine_codes(
            hidden_states[:, 2:3], frame_codes, cache=cache, temperature=0.0, refine=refine_frame
        )

    torch.testing.assert_close(
        refine_last(advanced_frames=2, skipped=False), refine_last(advanced_frames=1, skipped=True)
    )


def test_rows_left_out_of_refinement_do_not_disturb_the_refined_ones():
    refiner = _make_refiner(scale=6.0)
    hidden_states = torch.randn(2, 1, D_MODEL)
    pred_codes = _codes(batch_size=2, frames=1)

    both = refiner.refine_codes(hidden_states, pred_codes, temperature=0.0)
    one = refiner.refine_codes(hidden_states, pred_codes, temperature=0.0, refine=torch.tensor([True, False]))

    torch.testing.assert_close(both[0], one[0])
    assert (one[1] == MASK_TOKEN_ID).all()


def test_refine_rejects_a_row_mask_of_the_wrong_size():
    refiner = _make_refiner()
    with pytest.raises(ValueError, match="which rows to refine"):
        refiner.refine_codes(
            torch.randn(2, 1, D_MODEL), _codes(batch_size=2, frames=1), refine=torch.tensor([True, True, False])
        )


def test_advanced_context_reaches_the_frames_refined_after_it():
    # scaled up so an untrained stack actually reacts to its context
    frame_hidden = torch.randn(1, 1, D_MODEL)
    pred_codes = _codes(batch_size=1, frames=1)

    def refine_after(context_hidden):
        refiner = _make_refiner(scale=6.0)
        cache = _cache(refiner, batch_size=1)
        refiner.advance(context_hidden, cache=cache)
        return refiner.refine_codes(frame_hidden, pred_codes, cache=cache, temperature=0.0)

    context = torch.randn(1, 4, D_MODEL)
    torch.testing.assert_close(refine_after(context), refine_after(context.clone()))

    refined_codebooks = slice(_make_refiner().num_backbone_codebooks, None)
    assert not torch.equal(
        refine_after(context)[:, :, refined_codebooks], refine_after(context * -3.0)[:, :, refined_codebooks]
    ), "the advanced positions have to be read as left context"


def test_compute_loss_skips_frames_whose_targets_are_masked():
    """Prefilled positions ride along as context but carry no target, so they stay out of the loss."""
    refiner = _make_refiner()
    context, frames = 2, 4
    hidden_states = torch.randn(2, context + frames, D_MODEL)
    lengths = torch.tensor([context + frames] * 2)
    masked = torch.full((2, context, NUM_CODEBOOKS), MASK_TOKEN_ID)

    loss = refiner.compute_loss(hidden_states, torch.cat([masked, _codes(2, frames)], dim=1), lengths)
    assert torch.isfinite(loss) and loss > 0

    # with nothing but masked frames there is no prediction left to supervise
    all_masked = torch.full((2, context + frames, NUM_CODEBOOKS), MASK_TOKEN_ID)
    assert refiner.compute_loss(hidden_states, all_masked, lengths) == 0.0
