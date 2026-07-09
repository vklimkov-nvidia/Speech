# vLLM-Omni EasyMagpie Runtime Source

This repository initially promoted the verified `vllm_omni` source tree
recovered from `work/runner_package/work/vllm_omni` on 2026-06-29. The
`codex/magpie-vllm021-cfg-runtime` branch refreshes that tree from the exact
stable runtime staged at
`work/runner_package_prompt_streaming_pair_2n16g_v8/work/vllm_omni` on
2026-07-08. That runtime reports vLLM-Omni version `0.18.0` and runs against
vLLM `0.21.0` in image
`gitlab-master.nvidia.com:5005/vmendelev/work-kb/easymp-vllm-omni-original:20260612b`.

The chronological compatibility changes and EOS probes are recorded in
`work/patches/vllm_omni_vllm_compat_20260626.md` in the Magpie_optimized
project. The upstream commit was not embedded in either recovered source tree,
so the import and refresh commits are their durable source identities. No
import-time monkey patch is required: the tracked source package is placed
directly on `PYTHONPATH`.
