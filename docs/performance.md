# Performance notes

`python scripts/profile_engines.py` (Apple Silicon laptop, synthetic daily data, SMA
crossover, bps costs):

| Workload | Time |
|---|---|
| Vectorized, 10 instruments x 4,341 bars | 0.03 s |
| Vectorized, 100 instruments x 4,341 bars | 0.15 s |
| Vectorized, 500 instruments x 4,341 bars | 1.2 s |
| All metrics for one run | ~0.1 s |
| Event-driven, 10 instruments x 4,341 bars | 0.12 s |
| Event-driven, 100 instruments x 4,341 bars | 2.1 s |

The vectorized engine is dominated by round-trip trade extraction for large universes (one
pass per instrument). The event engine is a Python loop over bars by design. Use the
vectorized engine for sweeps and cross-sectional work and the event engine for
path-dependent logic. Research sweeps run trials in threads (`HFTF_MAX_WORKERS`).
