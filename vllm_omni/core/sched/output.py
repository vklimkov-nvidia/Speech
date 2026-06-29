from dataclasses import dataclass, field
from typing import Any

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request

from vllm_omni.engine import AdditionalInformationPayload


@dataclass
class OmniNewRequestData(NewRequestData):
    """New request data for omni models with embeddings support.

    Extends NewRequestData to include additional information for direct
    transfer between pipeline stages.

    Note: prompt_embeds is inherited from NewRequestData
    (torch.Tensor | None).

    Args:
        external_req_id: Optional external request ID for tracking
        additional_information: Optional serialized additional information
            dictionary containing tensors or lists
    """

    prompt_embeds: Any = None
    prefill_token_ids: list[int] | None = None
    external_req_id: str | None = None
    additional_information: AdditionalInformationPayload | None = None

    @property
    def mm_features(self):
        return getattr(self, "mm_kwargs", [])

    @mm_features.setter
    def mm_features(self, value):
        self.mm_kwargs = list(value or [])

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "OmniNewRequestData":
        """Create OmniNewRequestData from a Request object.

        Args:
            request: Request object to convert
            block_ids: Tuple of block ID lists for KV cache allocation
            prefill_token_ids: Optional prefill token IDs for v2 model runner

        Returns:
            OmniNewRequestData instance with data from the request
        """
        mm_kwargs = list(getattr(request, "mm_kwargs", None) or getattr(request, "mm_features", None) or [])
        mm_hashes = list(getattr(request, "mm_hashes", None) or [])
        mm_positions = list(getattr(request, "mm_positions", None) or getattr(request, "mm_placeholders", None) or [])
        fields = getattr(cls, "__dataclass_fields__", {})
        payload = dict(
            req_id=request.request_id,
            external_req_id=getattr(request, "external_req_id", None),
            prompt_token_ids=request.prompt_token_ids,
            mm_kwargs=mm_kwargs,
            mm_features=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            prefill_token_ids=prefill_token_ids,
            additional_information=getattr(request, "additional_information", None),
        )
        instance = cls(**{name: value for name, value in payload.items() if name in fields})
        for name, value in (
            ("mm_kwargs", mm_kwargs),
            ("mm_hashes", mm_hashes),
            ("mm_positions", mm_positions),
        ):
            if not hasattr(instance, name):
                setattr(instance, name, value)
        return instance


@dataclass
class OmniCachedRequestData(CachedRequestData):
    """Cached request data for omni models with embeddings support.

    Args:
        prompt_token_ids: Mapping from request ID to list of prompt token IDs
    """

    prompt_token_ids: dict[str, list[int]]
    additional_information: dict[str, dict | None]


@dataclass
class OmniSchedulerOutput(SchedulerOutput):
    """Scheduler output with omni-specific transfer metadata."""

    finished_requests_needing_kv_transfer: dict[str, dict] = field(default_factory=dict)
