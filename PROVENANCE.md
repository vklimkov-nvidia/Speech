# vLLM-Omni EasyMagpie Runtime Source

This repository promotes the verified `vllm_omni` source tree recovered from
`work/runner_package/work/vllm_omni` on 2026-06-29. The recovered package
reports vLLM-Omni version `0.18.0` and contains the EOS vLLM 0.10 compatibility
work used by prior successful EasyMagpie rollout runs.

The chronological compatibility changes and EOS probes are recorded in
`work/patches/vllm_omni_vllm_compat_20260626.md` in the Magpie_optimized
project. The upstream commit was not embedded in the recovered source tree, so
this repository's initial commit is the durable source identity for subsequent
campaign packages. No import-time monkey patch is required: the source package
is placed directly on `PYTHONPATH`.
