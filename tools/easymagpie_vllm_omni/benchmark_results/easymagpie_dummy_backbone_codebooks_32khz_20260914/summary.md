# EasyMagpie in-backbone codebook benchmark

## Run-4 results

The fourth measured run is the preselected comparison point. Acoustic codebook
sampling was enabled for every variant (`temperature=0.7`, `top_k=80`).

| Codebook tail | Concurrency | Requests/s | Mean TTFA | Mean ITL | Mean RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention + FFN | 1 | 1.50 | 64.7 ms | 35.4 ms | 15.12x |
| Attention + FFN | 32 | 10.29 | 274.4 ms | 168.0 ms | 103.73x |
| Mamba + FFN | 1 | 1.39 | 74.5 ms | 37.8 ms | 14.05x |
| Mamba + FFN | 32 | 10.19 | 284.2 ms | 172.1 ms | 102.73x |
| Attention only | 1 | 1.76 | 56.6 ms | 30.1 ms | 17.73x |
| Attention only | 32 | 10.79 | 263.0 ms | 159.8 ms | 108.75x |

Attention only was fastest at both concurrency levels. Relative to attention +
FFN, it delivered 17.3% more requests/s with 12.5% lower TTFA at concurrency 1,
and 4.9% more requests/s with 4.2% lower TTFA at concurrency 32. Relative to
Mamba + FFN, the corresponding improvements were 26.6% and 24.0% at concurrency
1, and 5.9% and 7.5% at concurrency 32.

## All measured runs

| Codebook tail | Run | c1 requests/s | c1 mean TTFA | c32 requests/s | c32 mean TTFA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention + FFN | 1 | 1.51 | 64.3 ms | 11.06 | 261.7 ms |
| Attention + FFN | 2 | 1.50 | 64.8 ms | 9.73 | 294.6 ms |
| Attention + FFN | 3 | 1.50 | 65.0 ms | 10.76 | 242.0 ms |
| Attention + FFN | 4 | 1.50 | 64.7 ms | 10.29 | 274.4 ms |
| Attention + FFN | 5 | 1.50 | 64.8 ms | 11.04 | 245.6 ms |
| Mamba + FFN | 1 | 1.40 | 74.3 ms | 10.03 | 292.0 ms |
| Mamba + FFN | 2 | 1.39 | 74.7 ms | 10.41 | 260.7 ms |
| Mamba + FFN | 3 | 1.39 | 75.1 ms | 10.45 | 269.3 ms |
| Mamba + FFN | 4 | 1.39 | 74.5 ms | 10.19 | 284.2 ms |
| Mamba + FFN | 5 | 1.39 | 74.9 ms | 10.24 | 268.4 ms |
| Attention only | 1 | 1.76 | 56.0 ms | 11.15 | 234.7 ms |
| Attention only | 2 | 1.75 | 56.7 ms | 10.35 | 268.2 ms |
| Attention only | 3 | 1.76 | 56.4 ms | 11.92 | 213.5 ms |
| Attention only | 4 | 1.76 | 56.6 ms | 10.79 | 263.0 ms |
| Attention only | 5 | 1.76 | 56.2 ms | 11.84 | 209.9 ms |

## Protocol

- GPU: NVIDIA RTX A6000
- Software: PyTorch 2.11.0+cu130, vLLM 0.24.0, vLLM-Omni 0.24.0
- Model: dummy weights, 32 backbone layers, 16-layer `MEMEM*EMEMEM*EME`
  prefix, followed by 16 per-codebook prediction layers
- Generation: 128 new tokens per request at 32 kHz
- Measured load: 128 requests per run at concurrency 1 and 32, repeated five
  times
- Warmup: 32 discarded requests at concurrency 1 and 32 for each freshly
  started architecture; the benchmark's built-in warmup also ran before each
  measured load
- Sampling: top-k Gumbel sampling for every acoustic codebook, with
  `temperature=0.7` and `top_k=80`
- Reproducibility: source commit `0df5c19cac5f0187cf4dc02747b1f3e96a4cd625`;
  uncommitted architecture implementation SHA-256
  `cb0bcc8a459b3f47dc498c10e7a446bf2645830472c5b3e9a0de47b194a4a57d`

All 3,840 measured requests completed successfully. Run 4 had one cumulative
playback underrun for attention + FFN at concurrency 32, and no cumulative
underruns for the other two variants. Attention only is the strongest candidate
for the separate no-sampling measurement.

See `metadata.txt`, `run_benchmarks.sh`, and the per-run logs in this directory
for the exact inputs and raw output.
