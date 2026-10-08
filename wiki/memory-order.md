# Read order changes timing even with a fixed address set

[Home](../README.md) · [Catalog](../data/catalog.json) · [Probe guide](../experiments/memory_order/README.md)

In this C550 experiment, changing read order produced substantial timing differences while holding the input address set, input span, logical read/write bytes and per-process buffer addresses fixed. For the same 64 MiB input, six processes with balanced order consistently observed a longer event-batch mean at `s=6` than at `s=12`. The within-process ratio was **3.284 [3.258, 3.288]**. Larger spacing between neighboring reads did not imply a longer time in this experiment.

A larger unique input set therefore cannot explain this control's difference. The experiment does not isolate cache reuse, address translation, transaction formation, memory partitioning or scheduling. The [complete samples and confirmation record](../data/results/20261008-memory-order.json) retain that scope.

## Why extend the original stride probe?

The [native strided gather](memory-access.md) holds output count fixed but increases input span with stride. That measures the cost of those gathers, but cannot isolate the effect of read order.

This experiment fixes `N` at `2^16`, `2^20` or `2^24` and tests `s=0,2,4,6,8,12` at each size. Every case reads each element of `input[0..N-1]` exactly once and writes consecutive `output[0..N-1]` elements. Each process allocates one input and one maximum-sized output buffer, reusing the same payload bases across cases. The frozen source construction and the mapping's bijection establish these address properties. No per-instruction address trace or physical DRAM placement was verified.

One kernel accepts runtime `N` and `s`, including `s=0`; it does not switch to a separate copy kernel. Output index `i` reads:

```text
k = log2(N)
j = ((uint64(i) << s) | (uint64(i) >> (k-s))) & (N-1)
output[i] = input[j]
```

The independent CPU oracle treats the input as a row-major `(N/2^s) × 2^s` matrix and flattens its transpose, using division and remainder to compute indices. For example, `N=8,s=2` reads `[0,4,1,5,2,6,3,7]`. This bijection preserves the unique address set and full span.

**Different s values produce different output permutations.** This is a parameterized access-pattern probe, not an unconditional replacement for the original gather. See the [memory_order README](../experiments/memory_order/README.md) for source, the mathematical mapping and checking procedure.

## Initial size sweep

Source `0e093ce` fixes block size at 256. Each case uses 10 warmups and 10 batches of 10 launches. These are median event-batch means in µs from the initial fixed-order sweep:

| s | 256 KiB input, N=2^16 | 4 MiB input, N=2^20 | 64 MiB input, N=2^24 |
| ---: | ---: | ---: | ---: |
| 0 | 8.422 | 17.101 | 186.317 |
| 2 | 8.474 | 17.651 | 227.072 |
| 4 | 8.832 | 21.786 | 746.957 |
| 6 | 9.114 | 34.816 | 1556.877 |
| 8 | 9.344 | 37.965 | 1270.310 |
| 12 | 9.382 | 36.979 | 474.522 |

Within a column, input/output address sets and logical traffic are fixed; columns differ in size. All final outputs of the initial 18 cases passed bitwise checking. Small cases may be strongly affected by submission and event boundaries. These values are not single-load latency or pure hardware access latency.

## Six-process confirmation of non-monotonic timing

For `N=2^24`, six predetermined orders place each s at each position once and each of the 30 directed adjacent pairs once. Each independent process completed device execution and release before full CPU checking. All 36 cases passed.

For each case, the analysis takes the median of ten batches within a process, then reports median [min, max] over six processes:

| s | Event-batch mean (µs) | Within-process T(s)/T(0) |
| ---: | ---: | ---: |
| 0 | 187.264 [186.022, 192.397] | 1.000 |
| 2 | 226.598 [225.830, 231.898] | 1.212 [1.205, 1.217] |
| 4 | 746.726 [745.779, 751.923] | 3.986 [3.908, 4.009] |
| 6 | 1556.736 [1555.546, 1561.741] | 8.314 [8.117, 8.362] |
| 8 | 1275.462 [1268.685, 1277.952] | 6.805 [6.642, 6.857] |
| 12 | 474.074 [473.062, 479.283] | 2.533 [2.491, 2.543] |

Ratios are computed within a process, not by dividing aggregate medians. The 60 batches are not 60 independent replications. Every process observed `s=6` slower than `s=12`. These are timing ratios between different read permutations, not speedups for the original operator.

## What the traces show and what remains unresolved

Separate traces covered the preselected `s=0` and `s=12` cases. A diagnostic `s=6` capture followed confirmation. All three runs passed final-output checks and contained 110 actual GPU kernel events each. They consistently reported 8 registers per thread and zero shared/private memory; every reported recompilation flag was false.

The observations showed no difference in the reported recompilation or resource-allocation paths. Equal register counts do not prove equal memory transactions or identify a unique bottleneck. Although the whole address set is fixed, **short-term reuse distance and the active working window still change**. These remain factors for further controls. Without hardware counters or discriminating counterexamples, the results do not identify cache-line size, bank count, TLB capacity or DRAM traffic.

The experiment uses one position-distinguishing finite FP32 input, not a random-input generalization set. Every N is divisible by block256, so no partial-block tails are covered. Default-stream event intervals can include host submission gaps. Runtime data-cache policy is unverified, clocks are not fixed and system-wide exclusivity is not claimed. The three profiled runs remain separate from ordinary timing, and raw trace units have not been converted.

A next implementation comparison can preserve **the same transpose result** while changing direct access to two-dimensional tiling, then test reuse windows and read/write organization. Such a comparison needs its own correctness, resource and timing evidence; this page alone establishes no implementation gain.

## Successor comparison with identical outputs

The [transpose tiling experiment](transpose.md) pursues that question with identical output semantics, including nonsquare and tail dimensions. It records whole-implementation gains separately from the smaller padding effects. It does not reuse this page's permutation ratios as implementation speedups.
