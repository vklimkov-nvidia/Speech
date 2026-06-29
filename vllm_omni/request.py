from collections.abc import Callable
import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.v1.request import Request, StructuredOutputRequest

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_omni.engine import AdditionalInformationPayload, OmniEngineCoreRequest, PromptEmbedsPayload


class OmniRequest(Request):
    """Request class for omni models, extending the base Request.

    This class extends the base vLLM Request with support for prompt
    embeddings and additional information payloads, enabling direct
    transfer of pre-computed embeddings between stages.

    Args:
        prompt_embeds: Optional serialized prompt embeddings payload.
            Used for direct transfer of embeddings between stages.
        additional_information: Optional additional information payload
            containing tensors or lists to be passed along with the request.
    """

    def __init__(
        self,
        prompt_embeds: PromptEmbedsPayload | torch.Tensor | None = None,
        # Optional external request ID for tracking
        external_req_id: str | None = None,
        additional_information: AdditionalInformationPayload | None = None,
        *args,
        **kwargs,
    ):
        prompt_embeds_tensor = self._maybe_decode_prompt_embeds(prompt_embeds)
        base_init = super().__init__
        try:
            base_signature = inspect.signature(base_init)
        except Exception:
            base_signature = None
        if base_signature is None:
            base_init(*args, **kwargs)
        else:
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in base_signature.parameters.values()
            )
            base_kwargs = dict(kwargs)
            if "prompt_embeds" in base_signature.parameters:
                base_kwargs["prompt_embeds"] = prompt_embeds_tensor
            if not accepts_kwargs:
                base_kwargs = {key: value for key, value in base_kwargs.items() if key in base_signature.parameters}
            base_init(*args, **base_kwargs)
        self.prompt_embeds = prompt_embeds_tensor
        self.mm_features = self.mm_kwargs
        # Preserve serialized prompt embeddings payload (optional)
        self.prompt_embeds_payload: PromptEmbedsPayload | None = (
            prompt_embeds if isinstance(prompt_embeds, PromptEmbedsPayload) else None
        )
        # Optional external request ID for tracking
        self.external_req_id: str | None = external_req_id
        # Serialized additional information payload (optional)
        self.additional_information: AdditionalInformationPayload | None = additional_information

    @staticmethod
    def _maybe_decode_prompt_embeds(
        prompt_embeds: PromptEmbedsPayload | torch.Tensor | None,
    ) -> torch.Tensor | None:
        if isinstance(prompt_embeds, PromptEmbedsPayload):
            dtype = getattr(np, prompt_embeds.dtype)
            arr = np.frombuffer(prompt_embeds.data, dtype=dtype)
            arr = arr.reshape(prompt_embeds.shape)
            return torch.from_numpy(arr)
        return prompt_embeds

    @classmethod
    def from_engine_core_request(
        cls,
        request: OmniEngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        """Create an OmniRequest from an OmniEngineCoreRequest.

        Args:
            request: The OmniEngineCoreRequest to convert
            block_hasher: Optional function to compute block hashes for
                prefix caching

        Returns:
            OmniRequest instance created from the engine core request
        """
        mm_kwargs: Any = getattr(request, "mm_kwargs", None)
        if mm_kwargs is not None:
            mm_kwargs = list(mm_kwargs)
        elif getattr(request, "mm_features", None) is not None:
            mm_kwargs = list(getattr(request, "mm_features"))

        sampling_params = getattr(request, "sampling_params", None)
        structured_output_request = StructuredOutputRequest(sampling_params=sampling_params) if sampling_params else None

        return cls(
            request_id=request.request_id,
            # Optional external request ID for tracking
            external_req_id=getattr(request, "external_req_id", None) or request.request_id,
            client_index=getattr(request, "client_index", 0),
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            multi_modal_kwargs=mm_kwargs,
            multi_modal_hashes=getattr(request, "mm_hashes", None),
            multi_modal_placeholders=getattr(request, "mm_placeholders", None),
            sampling_params=sampling_params,
            pooling_params=getattr(request, "pooling_params", None),
            eos_token_id=getattr(request, "eos_token_id", None),
            arrival_time=getattr(request, "arrival_time", None),
            lora_request=getattr(request, "lora_request", None),
            structured_output_request=structured_output_request,
            cache_salt=getattr(request, "cache_salt", None),
            priority=getattr(request, "priority", 0),
            block_hasher=block_hasher,
            additional_information=getattr(request, "additional_information", None),
        )
