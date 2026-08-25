# Qwen3.8-27B scaling grid — RTX PRO 6000, TG 300 (132-500)

### Aggregate output tok/s

| PP | quant | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| 512 | NVFP4 | 145.2 | 287.1 | 539.6 | 949.2 | 1454.7 | 1880.8 | 2375.6 |
| 2K | NVFP4 | 129.5 | 239.1 | 422.5 | 690.0 | 955.8 | 1199.8 | 1270.2 |
| 4K | NVFP4 | 121.6 | 214.5 | 356.0 | 524.1 | 661.9 | 765.6 | 752.6 |
| 8K | NVFP4 | 106.1 | 175.8 | 261.5 | 334.2 | 391.7 | 409.0 | 423.7 |
| 10K | NVFP4 | 101.5 | 167.4 | 236.5 | 284.3 | 311.9 | 333.5 | 324.2 |

### Per-stream output tok/s (1000/TPOT)

| PP | quant | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| 512 | NVFP4 | 143.5 | 150.2 | 138.0 | 122.6 | 93.2 | 65.4 | 37.6 |
| 2K | NVFP4 | 119.3 | 114.4 | 105.3 | 85.2 | 58.6 | 36.7 | 19.0 |
| 4K | NVFP4 | 114.7 | 107.0 | 87.0 | 65.1 | 40.4 | 22.9 | 9.9 |
| 8K | NVFP4 | 100.7 | 87.4 | 66.3 | 41.6 | 22.0 | 13.1 | 6.5 |
| 10K | NVFP4 | 96.7 | 83.4 | 56.9 | 35.6 | 18.1 | 10.4 | 5.1 |

### Request latency p50 (s)

| PP | quant | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| 512 | NVFP4 | 2.1 | 2.1 | 2.3 | 2.5 | 3.3 | 4.7 | 8.1 |
| 2K | NVFP4 | 2.2 | 2.6 | 2.8 | 3.5 | 5.0 | 8.4 | 15.5 |
| 4K | NVFP4 | 2.5 | 2.7 | 3.6 | 4.7 | 7.3 | 13.2 | 28.2 |
| 8K | NVFP4 | 2.9 | 3.4 | 4.6 | 7.1 | 12.3 | 23.0 | 46.0 |
| 10K | NVFP4 | 3.0 | 3.8 | 5.5 | 8.1 | 14.6 | 29.1 | 60.2 |

### Long context — aggregate output tok/s

| PP | quant | 1 | 4 | 8 | 10 | 16 |
|---|---|---|---|---|---|---|
| 100K | NVFP4 | 16.1 | 19.9 | 21.1 | — | 17.4 |
| 150K | NVFP4 | 9.7 | 11.3 | 12.5 | 10.8 | — |

### Long context — request latency p50 (s)

| PP | quant | 1 | 4 | 8 | 10 | 16 |
|---|---|---|---|---|---|---|
| 100K | NVFP4 | 17.8 | 65.1 | 137.9 | — | 263.5 |
| 150K | NVFP4 | 32.6 | 128.2 | 273.6 | 271.3 | — |

### Long context — per-stream tok/s

| PP | quant | 1 | 4 | 8 | 10 | 16 |
|---|---|---|---|---|---|---|
| 100K | NVFP4 | 17.0 | 4.5 | 2.4 | — | 1.2 |
| 150K | NVFP4 | 10.1 | 2.2 | 1.1 | 1.1 | — |

### Long context — TOTAL tok/s (prompt + output)

This is the row that says what the card is really doing: at 100K prompts the generated 300 tokens are ~0.3% of the work.

| PP | quant | 1 | 4 | 8 | 10 | 16 |
|---|---|---|---|---|---|---|
| 100K | NVFP4 | 5903 | 6466 | 6609 | — | 6049 |
| 150K | NVFP4 | 5041 | 5300 | 5519 | 5533 | — |

### Long context — completed requests/min

| PP | quant | 1 | 4 | 8 | 10 | 16 |
|---|---|---|---|---|---|---|
| 100K | NVFP4 | 3.97 | 3.81 | 3.17 | — | 3.17 |
| 150K | NVFP4 | 2.06 | 1.75 | 1.27 | 1.27 | — |

### Achieved vs requested concurrency

| PP | quant | 1 | 4 | 8 | 10 | 16 |
|---|---|---|---|---|---|---|
| 100K | NVFP4 | 1.0 | 3.8 | 6.6 | — | 12.2 |
| 150K | NVFP4 | 1.0 | 3.6 | 5.6 | 5.6 | — |

### Errors — prompt-length grid

None. Every level of every scenario completed with zero failed requests.

### Errors — long context

None. Every level of every scenario completed with zero failed requests.
