# C500 Triton optimization lessons from a 61-task training round

[Home](../README.md) · [Method](../docs/methodology.md) · [Catalog](../data/catalog.json)

These observations come from one C500 (xcore1000) development round on MACA 3.7.0.38, retained
outside this repository on the C500 host. Each task kept its original workload, oracle and ABI, and
timed candidates with paired MCPTI kernel timing against a fixed per-task baseline. The retrieval
label is `source-reported / external-observation`: the numbers below were re-read from the retained
receipts on 2026-10-28, but they are not measurements made by this repository's probes, and a C500
result is not a C550 measurement.

<!-- kernelwiki:summary:start -->
## What to try and when

- For FP32 GEMM-like kernels (`R=1024, K=256, N=64`), try `tl.dot` with `input_precision="ieee"` on a row tile of 32 and a column tile of 32 with 4 warps. It beat the FP32 SIMT baseline about 25 times, from 0.264 ms to 0.0102 ms. Row tiles of 16 were refused.
- Consume `B` as stored (`[K, N]`) in `tl.dot`; do not transpose it first. Fuse simple epilogues such as bias and sigmoid into the same kernel.
- For a decode-style attention at the same shape, reusing one K/V block across a block of query rows helped. Splitting the output columns over 2 programs helped more (0.203 ms against 0.263 ms unsplit). A 4-way split gained nothing further.
- Expect a timing floor of about 2.56-2.82 us. Many elementwise and small reduction tasks reached it, and then width, `tl.fma`, `tl.cat` or alignment hints changed nothing measurable.
- Budget 16-warp launches as risky: in this round they failed at launch in 23 tasks with `mcErrorRecompile`. 8 warps was the practical ceiling for these kernels.
- A per-thread register or local footprint can exceed the launch limit (`status 32`, `mcErrorMemoryValueTooLarge`). Large whole-tile register arrays and 128-256-wide column tiles hit it in 7 tasks.

These are conditional search suggestions for the stated shapes and SDK. They establish no default tile, no
C550 behaviour and no general ranking against vendor libraries.
<!-- kernelwiki:summary:end -->

## Matrix-core path for FP32 GEMM

The `gemm` and `gemm_silu` tasks compute `out = a @ b + bias` (with a SiLU epilogue in the second) for
`a[R,K]`, `b[K,N]`, FP32, primary case `R=1024, K=256, N=64`. Every faster candidate used `tl.dot` with IEEE
FP32 input precision on 2-D tiles:

| Tiling (rows × columns, K block, stages, warps) | Candidate | Fixed baseline |
|---|---:|---:|
| 32 × 32, K32, 2 stages, 4 warps | 0.01024 ms | 0.264 ms |
| 32 × 32, K128, 2 stages, 4 warps, no mask | 0.01075 ms | 0.265 ms |
| 64 × 64, K32, 2 stages, 8 warps | 0.01613 ms | 0.265 ms |
| `gemm_silu`, 32 × 32, 4 warps, fused sigmoid | 0.01075 ms | 0.264 ms |

All were correctness-accepted on every task-owned case.

- **Tiles of 16 rows:** several of these were rejected. A 16-row × 128-column tile with K64 hit the per-thread launch limit described below.
- **Tile size:** larger 64 × 64 tiles were slower than 32 × 32 at this size.
- **Epilogue:** fusing the epilogue into the same kernel kept the 32 × 32 time.
- **Transposes:** taking `B` as stored needed no explicit transpose. A transpose-first variant gave no benefit.
- **Precision:** use the IEEE input-precision setting the oracle needs. A lower-precision dot is a different numerical contract and must pass the original tolerance on its own.

## Row-block reuse for decode attention

`attention_decode` (`q[R,K]`, `k,v[N,K]`, FP32, `R=1024, K=256, N=64`):

| Variant | Time | Relative to fixed baseline (0.354 ms) |
|---|---:|---:|
| Kernel that loads one K/V block and applies it to a block of query rows | 0.263 ms | 1.34× |
| Same kernel, output columns split over 2 programs (grid 128) | 0.203 ms | 1.74× |
| Same kernel, output columns split over 4 programs (grid 256) | 0.212 ms | 1.67× |

The split adds programs without re-reading K/V per row. The gain stopped at 2. With `N=64`, the key set fits
one tile, so an online-softmax rescaling loop has nothing to amortize at this shape.

## The timing floor

Across 851 correctness-passing candidate timings, the smallest median was 2.56 us. The most common values
were 2.816, 3.072, 4.096, 4.864 and 5.12 us, so the reported medians are quantized. Tasks that reached about
2.8 us did not improve further with:

- more or fewer warps;
- 16-byte alignment hints (`tl.multiple_of`, `tl.max_contiguous`): unchanged at 4.096 us on `bias_gradient_reduction`;
- an explicit `tl.fma` fold: unchanged, 0.00512 ms on both sides;
- stacking three row reductions with `tl.cat`: unchanged.

Before tuning such a kernel, compare its median with this floor. Once it is within one or two timer steps,
host-side cost or the number of launches is the remaining lever.

## Launch limits seen at run time

- **16 warps:** launches with `num_warps=16` failed with `mcErrorRecompile` from `mcModuleLaunchKernel` in 23
  tasks. Fifteen other 16-warp candidates did launch and pass, so this is not a fixed rule, but the failure
  was common enough to treat 16 warps as an expensive exploration point.
- **Local memory:** launches failed with status 32 (`mcErrorMemoryValueTooLarge`) in 7 tasks. Triggers were
  wide column tiles (128-512), 32-row tiles, a 16 × 128 MMA tile, and a whole-tile `[4,64,256]` register
  array that needed 4312 bytes of local memory per thread. The error is reported at launch, after
  compilation succeeded. Shrink the per-thread tile, or move the reused operand into a K loop, rather than
  adding warps.

## Things that did not pay at these shapes

- **Grouping rows into one program:** for row-wise normalizations (RMSNorm, LayerNorm and residual
  variants, 128-1024 rows), grouping 2-16 rows was slower than one row per program, typically 0.4-0.8×.
- **Splitting a row reduction across programs:** for cross-row reductions, this needs a second kernel or
  atomics, and at about 5 us per kernel the extra launch removes any gain.
- **Pairwise distance:** row-group variants measured between 0.066 and 0.557 ms, all slower than the
  best non-grouped candidate.

## Limitations

- One device, one SDK, one shape per task; no sweep locates crossover points.
- Paired timing is against each task's fixed baseline, not between two optimized variants.
- Launch failures are counted by task, not by attempt, and the round did not isolate whether a
  recompile failure depends on registers, shared memory or the generated binary.
- The original round artifacts stay on the C500 host; this page keeps only the bounded reading above.
