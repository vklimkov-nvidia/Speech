# Parallel Codebook Dummy Benchmark

The talker uses `codebook_prediction_mode="parallel"` with
`local_transformer_n_layers=0`. One direct linear projection maps each
backbone hidden state to logits for all 16 stacked codebooks, and all codebooks
are sampled in one compiled graph.

Each run measured 128 requests at concurrency 1 and 32 after the benchmark's
built-in warmup. Both stages used dummy weights, generation was fixed at 128
acoustic steps, and output used the 32 kHz codec architecture.

## Per-Run Results

| Run | C=1 requests/s | C=1 mean TTFA | C=32 requests/s | C=32 mean TTFA |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.93 | 56.6 ms | 11.14 | 271.2 ms |
| 2 | 1.92 | 56.7 ms | 10.25 | 266.9 ms |
| 3 | 1.92 | 56.4 ms | 10.05 | 287.5 ms |
| 4 | 1.92 | 56.8 ms | 11.28 | 220.5 ms |
| 5 | 1.92 | 56.9 ms | 11.60 | 225.0 ms |

All 1,280 measured requests succeeded. Runs 1 and 3 each recorded one playback
underrun at concurrency 32; the other measured levels recorded none.

## Run 4 Comparison

| Local transformer | Concurrency | Requests/s | Mean TTFA |
| --- | ---: | ---: | ---: |
| 3 layers | 1 | 0.76 | 110.3 ms |
| 1 layer | 1 | 1.19 | 78.5 ms |
| disabled, parallel heads | 1 | 1.92 | 56.8 ms |
| 3 layers | 32 | 6.03 | 390.2 ms |
| 1 layer | 32 | 8.61 | 287.7 ms |
| disabled, parallel heads | 32 | 11.28 | 220.5 ms |

Relative to the 3-layer architecture in run 4, parallel prediction increased
requests/s by 152.6% at concurrency 1 and 87.1% at concurrency 32, while mean
TTFA fell by 48.5% and 43.5%, respectively.

Relative to the 1-layer architecture in run 4, parallel prediction increased
requests/s by 61.3% at concurrency 1 and 31.0% at concurrency 32, while mean
TTFA fell by 27.6% and 23.4%, respectively.
