# SOCK before/after benchmark

| case | c | baseline TMH tok/s | post-fix TMH tok/s | TMH change | paired gap change | token mean before→after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| extended_generation_768 | 1 | 121.38 | 123.76 | +1.97% | +1.59 pp | 768.00→768.00 |
| extended_generation_768 | 2 | 196.20 | 201.81 | +2.86% | +1.87 pp | 768.00→768.00 |
| extended_generation_768 | 4 | 226.69 | 243.06 | +7.22% | +2.78 pp | 768.00→768.00 |
| long_context_summary_256 | 1 | 108.90 | 113.39 | +4.12% | +2.99 pp | 256.00→256.00 |
| long_context_summary_256 | 2 | 147.41 | 164.06 | +11.29% | +5.62 pp | 256.00→256.00 |
| long_context_summary_256 | 4 | 129.72 | 155.98 | +20.24% | +4.47 pp | 256.00→256.00 |
| long_cosmology_512 | 1 | 125.95 | 126.65 | +0.55% | +0.46 pp | 512.00→512.00 |
| long_cosmology_512 | 2 | 206.47 | 207.59 | +0.54% | +0.37 pp | 512.00→512.00 |
| long_cosmology_512 | 4 | 276.46 | 282.06 | +2.03% | +0.94 pp | 512.00→491.67 |
| medium_architecture_256 | 1 | 130.06 | 129.29 | -0.59% | -0.55 pp | 256.00→256.00 |
| medium_architecture_256 | 2 | 219.92 | 219.06 | -0.39% | -0.31 pp | 256.00→256.00 |
| medium_architecture_256 | 4 | 325.55 | 324.28 | -0.39% | -0.21 pp | 256.00→256.00 |
| short_codegen_128 | 1 | 130.13 | 131.17 | +0.79% | +0.69 pp | 128.00→128.00 |
| short_codegen_128 | 2 | 225.93 | 224.65 | -0.56% | -4.28 pp | 128.00→128.00 |
| short_codegen_128 | 4 | 343.42 | 340.19 | -0.94% | -0.68 pp | 128.00→128.00 |
| tiny_fact_64 | 1 | 133.57 | 133.19 | -0.29% | -0.28 pp | 64.00→64.00 |
| tiny_fact_64 | 2 | 235.83 | 232.39 | -1.46% | -1.22 pp | 64.00→64.00 |
| tiny_fact_64 | 4 | 382.13 | 376.16 | -1.56% | -1.03 pp | 64.00→64.00 |

Median TMH throughput change: +0.55%.
Median standard-control change: +0.00%.
Median paired-gap change: +0.42 percentage points.
Post-fix token mean met or exceeded baseline in 17/18 cells.
