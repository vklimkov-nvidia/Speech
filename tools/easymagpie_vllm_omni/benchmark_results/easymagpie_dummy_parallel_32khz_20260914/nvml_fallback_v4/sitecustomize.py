"""Select CUDA platforms when host NVML is temporarily unavailable."""

try:
    import vllm.platforms
except ModuleNotFoundError:
    # ``conda run`` starts under the base interpreter before launching the
    # requested environment, and vLLM is intentionally absent there.
    pass
else:

    def _non_nvml_cuda_platform() -> str:
        return "vllm.platforms.cuda.CudaPlatform"

    vllm.platforms.builtin_platform_plugins["cuda"] = _non_nvml_cuda_platform
    # vLLM resolves once during its own package import, before this startup
    # hook can replace the detector. Clear that early unspecified result.
    vllm.platforms._current_platform = None

    try:
        import vllm_omni.platforms
    except ModuleNotFoundError:
        pass
    else:

        def _cuda_omni_platform() -> str:
            return "vllm_omni.platforms.cuda.platform.CudaOmniPlatform"

        vllm_omni.platforms.builtin_omni_platform_plugins["cuda"] = _cuda_omni_platform
        vllm_omni.platforms._current_omni_platform = None
