# EasyMagpie grouped in-backbone codebook benchmark

## Run-4 results

The fourth measured run is the preselected comparison point. Acoustic codebook
sampling was enabled (`temperature=0.7`, `top_k=80`).

| Concurrency | Requests/s | Mean TTFA | Mean ITL | Mean RTF | Successful requests |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.39 | 74.4 ms | 38.1 ms | 13.97x | 128/128 |
| 32 | 10.52 | 272.6 ms | 164.3 ms | 106.00x | 128/128 |

Run 4 had no deadline misses or cumulative playback underruns at either
concurrency level.

## Architecture comparison

| Architecture | c1 requests/s | c1 mean TTFA | c32 requests/s | c32 mean TTFA |
| --- | ---: | ---: | ---: | ---: |
| 16-prefix attention + FFN, one code per block | 1.50 | 64.7 ms | 10.29 | 274.4 ms |
| 16-prefix attention only, one code per block | 1.76 | 56.6 ms | 10.79 | 263.0 ms |
| 31-prefix grouped 3x4 attention + FFN | 1.39 | 74.4 ms | 10.52 | 272.6 ms |

Compared with the earlier attention+FFN tail, the grouped model has 7.3% lower
throughput and 15.0% higher TTFA at concurrency 1. At concurrency 32 it has 2.2%
higher throughput and 0.7% lower TTFA. The grouped model retains all 31 original
backbone entries, so this is an end-to-end architecture comparison rather than
an equal-depth tail comparison.

## All measured runs

| Run | c1 requests/s | c1 mean TTFA | c32 requests/s | c32 mean TTFA |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.39 | 74.6 ms | 10.42 | 259.5 ms |
| 2 | 1.38 | 74.6 ms | 10.15 | 283.3 ms |
| 3 | 1.38 | 74.6 ms | 10.81 | 263.0 ms |
| 4 | 1.39 | 74.4 ms | 10.52 | 272.6 ms |
| 5 | 1.39 | 74.4 ms | 10.34 | 268.3 ms |

All 1,280 measured requests completed successfully. Run 1 had eight cumulative
playback underruns and nine deadline misses at concurrency 32. Runs 2 through 5
had no cumulative underruns or deadline misses.

## Protocol

- GPU: NVIDIA RTX A6000
- Software: PyTorch 2.11.0+cu130, vLLM 0.24.0, vLLM-Omni 0.24.0
- Original backbone: 31 entries,
  `MEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`
- Prediction tail: four sequential groups, each containing three composite
  attention+dense-FFN blocks and four parallel codebook heads
- Logical config: 43 layer entries (31 original + 12 composite tail blocks);
  the composite tail executes 24 native sublayers
- Feedback: each non-final group averages its four sampled code embeddings,
  projects the result, and adds it to the live hidden/residual stream
- Generation: 128 new tokens per request at 32 kHz
- Measured load: 128 requests per run at concurrency 1 and 32, repeated five
  times
- Warmup: 32 discarded requests at concurrency 1 and 32 after fresh service
  startup; the benchmark's built-in warmup also ran before each measured load
- Sampling: top-k Gumbel acoustic sampling with `temperature=0.7` and `top_k=80`
- Base commit: `506f7d034c7353414bc2af83dea50762c637096a`
- Uncommitted implementation SHA-256:
  `94e40003b50e7a398db824aa945e488ea33d73aae28738e73df770af52fe23fa`

See `metadata.txt`, `run_benchmarks.sh`, and the per-run logs in this directory
for the exact inputs and raw output.
