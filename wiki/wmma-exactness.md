# Native C550 WMMA: failed exact acceptance and reproducible residuals

[Home](../README.md) · [Catalog](../data/catalog.json) · [Probe guide](../experiments/wmma/README.md)

In this C550 / MACA 3.5.3.18 / MXCC `1.0.0 (6477545d4d)` environment, native 16×16×16 FP16 WMMA with a float accumulator fragment compiles and executes, but **does not satisfy the strict exact numerical contract for these inputs**. Prepared inputs are valid, guards are intact and all outputs are finite. Scalar controls are exact. The [fixed-result magnitude scan](#successor-adjacent-product-magnitudes-at-a-fixed-result) tests products `(q, -(q+1))/256` for every integer q from 1 through 14. Their exact sum is always -1/256: pairs at q1–6 are exact, while q7–14 each have one C00 residual of -2^-31. This is a bounded input-grid observation, not a universal threshold or an identified arithmetic mechanism. The [A-scaling study](#successor-exact-power-of-two-scaling-of-a) finds a constant normalized residual at q7/q12. The latest [reciprocal-exponent control](#successor-reciprocal-exponents-with-fixed-products) holds each product and reference fixed: complete outputs remain unchanged across its five exponent pairs, with q6 exact and q7/q12 retaining a signed residual of -2^-31.

## Reading guide

| Question | Evidence |
| --- | --- |
| What is the exact numerical contract? | [Inputs and oracle](#predetermined-matrix-and-numerical-contract) · [Error metric](#define-the-error-metric) |
| What failed first, and what does the API promise? | [Initial results](#initial-results-and-independent-reproduction) · [Documentation and profiling](#what-documentation-and-profiling-establish) |
| Do captured inputs and a scalar control agree? | [Input snapshots and scalar results](#successor-device-input-snapshots-and-a-scalar-fp32-source-control) |
| Can the residual be reproduced with two products? | [K-prefix scan](#successor-logical-k-prefixes-and-a-two-term-witness) · [Isolation and relocation](#successor-isolation-and-relocation-of-the-two-products) |
| What changes with placement, signs or magnitudes? | [Components and K slots](#successor-single-products-and-k-slot-permutations) · [Sign configurations](#successor-fixed-magnitude-sign-configurations) · [Fixed-result magnitude scan](#successor-adjacent-product-magnitudes-at-a-fixed-result) |
| Does exact scaling change the numerical response? | [A-only scaling](#successor-exact-power-of-two-scaling-of-a) · [Reciprocal exponents at fixed products](#successor-reciprocal-exponents-with-fixed-products) |

In the initial sweep, the first mismatch is C[0,0] for 16×16×16: reference `-0.5` (`0xbf000000`), observed `-0.5000000596046448` (`0xbf000001`). Two independent processes and a diagnostic trace using the same frozen binary reproduce the complete original output words. The [raw inputs, outputs and diagnostic record](../data/results/20261008-wmma-exact-diagnostic.json) explicitly retain `correctness.passed=false` and `performance_accepted=false`. No tolerance is relaxed and no performance conclusion is drawn from the timings.

## Predetermined matrix and numerical contract

Source `ef22b51` uses native `-x maca -offload-arch=xcore1000`, includes `mcr/mc_runtime.h` and the installed compiler resource header `__clang_maca_mma_functions.h`, and calls `mxmaca::wmma`. This is the actual installed route. The official example's `mma.h` is under the cu-bridge include path; its macro conditions do not by themselves qualify a native include route.

One complete 64-thread block handles one 16×16 output tile. A is packed row-major and B column-major into K chunks of width 16. Each operand stores four chunks; host preparation fills positions outside logical M/N/K with +0. Every thread follows the same `ceil(K/16)` loop, starts its float accumulator fragment at zero, calls the installed four-argument `mma_sync` and stores all 256 values row-major. K=0 still fills and stores the fragment without an early exit. Packing and transfers are excluded from event timing.

Logical inputs are:

```text
a_num(i,k) = ((67*i + 13*k) % 31) - 15
b_num(k,j) = ((17*k + 5*j + 3) % 29) - 14
A = a_num / 16; B = b_num / 16
C_ref(i,j) = sum_k a_num(i,k)*b_num(k,j) / 256
```

These inputs are exactly representable in binary16, and each product is an integer divided by 256. For K≤64, the absolute numerator of any subset or partial sum is bounded by `64×15×14=13440`, below 2²⁴. A sequence of correctly rounded FP32 multiplications and additions can therefore represent these intermediates exactly. Reassociating this ordinary sum does not itself require a residual.

The independent CPU oracle uses logical i/j/k integer dot products, not WMMA or fragment lane mappings. Acceptance requires all 256 outputs to be finite and numerically exact, treating +0/−0 as equal. It checks 64 guards on each side bitwise. CPU negative controls include incorrect B layout, omitted K chunks, unwritten padding, nonfinite outputs and outputs incorrectly rounded to FP16. The [implementation and reproduction commands](../experiments/wmma/README.md) retain the full contract.

## Initial results and independent reproduction

The following all-case diagnostic examines the retained initial outputs. Each case checks the entire 16×16 region, including zeros outside logical M/N:

| M×N×K | Outputs failing exact equality | Maximum absolute residual |
| --- | ---: | ---: |
| 16×16×0 | 0 | 0 |
| 1×1×1 | 0 | 0 |
| 16×16×16 | 30 | 5.96046448e-08 |
| 15×16×16 | 28 | 5.96046448e-08 |
| 16×15×16 | 28 | 5.96046448e-08 |
| 15×15×15 | 25 | 5.96046448e-08 |
| 7×9×17 | 3 | 5.96046448e-08 |
| 15×16×31 | 19 | 1.1920929e-07 |
| 16×15×32 | 18 | 1.1920929e-07 |
| 16×16×33 | 15 | 1.1920929e-07 |
| 9×7×63 | 1 | 4.76837158e-07 |
| 16×16×64 | 10 | 4.76837158e-07 |

The original strict CLI stopped at the first mismatch in the third case. The original plan's reverse run and performance traces were not started. A separate **failure-reproduction diagnostic** then retained the original `ef22b51` binary, inputs, oracle, warmups and timing settings: two independent dense K16 runs, one dense K64 run and one K16 diagnostic trace. Their A/B files and every output word, including guards, match the corresponding initial cases. The strict checker continues to fail. New runs did not overwrite the original failure.

The independent diagnostic also visits outputs that the first CLI never reached. Across all 16 actual cases, it retains 32,768 input halfwords, 4,096 C values and 2,048 guards. All retained inputs match the declared packing, all C values are finite and all guards are intact. Reference-zero and logical-padding positions remain exactly zero. A CPU replay of the actual inputs, rounding every product and accumulation to FP32, also reproduces the exact reference.

These checks rule out host-packing mistakes and ordinary FP32 summation reordering as explanations under the declared input contract; they do not locate a hardware defect. At this stage, device A/B readbacks and a GPU scalar-FP32 control had not been collected. The successor below adds both controls without uniquely attributing the cause to hardware.

## Define the error metric

The maximum absolute residual is `4.76837158203125e-7`. The largest adjacent-FP32 distance occurs at C[0,9] for M16/N15/K32:

| | Reference | Observed |
| --- | --- | --- |
| Value | 0.00390625 | 0.0039062313735485077 |
| FP32 bits | `0x3b800000` | `0x3b7fffb0` |

A monotonic FP32 encoding counts **adjacent representable-value steps**, treating both signed zeros as one value. The distance is 80 steps. The reference lies exactly on a power-of-two boundary; dividing by the spacing upward from the reference instead gives 40. Calling this simply 80 ULP without defining the convention would be ambiguous. It is an observed bound for the tested data, not a general tolerance recommendation.

## What documentation and profiling establish

The [official 3.5.3.x C++ guide's WMMA section][wmma] describes D=A×B+C, fragment types and full-warp participation. Its [type table][types] includes FP16 inputs with a float accumulator fragment. The reviewed section provides no identified guarantee for intermediate precision, rounding mode, error bounds or equivalence to stepwise IEEE FP32 multiply-add. Rounding clauses for half/half2 arithmetic or conversion elsewhere on the page cannot be transferred to WMMA. The current conclusion is therefore **failure of the strong exact contract with an unexplained numerical mechanism**, not a demonstrated vendor defect.

The diagnostic trace contains 110 `wmma_tile_kernel` events. It reports 28 registers, zero static/dynamic shared memory and zero private memory per thread, with 110 false recompilation flags and no missing values. This does not identify the arithmetic mechanism behind the residual. Raw event batches and trace durations remain only for traceability. Trace units are independently unverified; no throughput, speedup or per-MMA latency is reported. Five device workers and one profiled application exited and passed release observations.

This page is indexed as `locally-measured / device-correctness` to describe a direct numerical diagnostic. The performance scope `local-measurement` still requires a complete passing correctness check. Indexing a failure does not turn it into a pass.

A successor uses new source and a new diagnostic plan for input readbacks and a scalar control, preserving the original failure. None of these studies promoted a change to the open-cake-ir Compiler, Target or calibration.

<a id="后继设备输入快照与标量fp32源码对照"></a>

## Successor: device input snapshots and a scalar FP32 source control

Source `403a74a` adds a separate control mode while preserving the WMMA function body. This does not assert that the compiled binary is identical to `ef22b51`. Each logical case uploads inputs once to the same A/B device allocations, then runs WMMA and scalar in its declared order. Each implementation has its own C allocation, guards and fixed output filename. Each complete output is saved immediately after its implementation finishes; the first output is saved before the second implementation runs.

The scalar source launches 256 threads, one per C element. It uses `__half2float` to read the same packed A/B addresses, traverses the same complete K chunks including padding, and performs multiplication and addition with float variables. WMMA uses 64 threads and the fragment interface. Launch geometry and load code differ: this is a functional control. Ordinary float source does not prove separate scalar machine instructions or forbid FMA contraction.

Complete A/B readbacks occur before the first computation, between implementations and after the second computation. Every word in every snapshot is checked against the original input file and fixed packing contract. There is no A/B rewrite between implementations.

| Paired run | Logical cases | Unequal WMMA elements | Unequal scalar elements | Three-stage input readbacks |
| --- | ---: | ---: | ---: | --- |
| Forward cases, WMMA first | 12 | 177 | 0 | All match |
| Reverse cases, scalar first | 12 | 177 | 0 | All match |
| Paired K16 trace, WMMA first | 1 | 30 | 0 | All match |

All 25 scalar 16×16 results exactly match the original integer-dot/256 oracle. WMMA passes only the repeated K0 and 1×1×1 shapes; the other 21 output matrices still contain mismatches. Every output is finite and every guard is intact. All 153,600 device-readback halfwords match the input contract. They form 75 paired A/B snapshots, not 75 different random inputs.

For 16×16×16, C[0,0] remains the same in both orders and the paired trace:

| Path | Observed value | Observed FP32 bits |
| --- | ---: | --- |
| WMMA | -0.5000000596046448 | `0xbf000001` |
| Scalar source | -0.5 | `0xbf000000` |

The successor's default mode0 binary also retains the original strict failure behavior across its 12 regression cases. Direct comparison of all 37 new WMMA outputs, comprising 12 default and 25 paired outputs, finds every complete output word equal to the corresponding `ef22b51` observation. Equality is an observation, not a condition imposed by the projection. Scalar success does not overwrite WMMA failure.

The [control record](../data/results/20261008-wmma-scalar-control.json) publishes prepared inputs, all three readback stages and both C outputs. Default regression plus paired runs contain 62 variant outputs, 15,872 payload words, 7,936 guards and 620 raw timing batches. The overall contract remains `passed=false` and `performance_accepted=false`. The same oracle retains failures; timing is not compared to claim a speedup.

The paired trace has exactly 220 kernel events: 110 WMMA and 110 scalar events. Each group identifies its own first 10 warmups; no single global prefix is removed. The tool reports WMMA block64/28 registers and scalar block256/36 registers. Both groups report zero shared/private memory and 110 false recompilation flags. All four workers and one profiled application exited and passed release observations.

For this SDK, input and instrumentation, the scalar source path produces exact results from the shared device inputs while WMMA reproduces the original residuals. This narrows the investigation to the WMMA execution path under these conditions. Three snapshots establish state only at their capture boundaries; they do not exclude transient values inside the kernel, specialized fragment loads, different lowering or internal arithmetic. They do not uniquely attribute the cause to hardware.

The successor below reduces logical matrix extents and K prefixes while preserving the physical tile and scalar control. Native instruction behavior remains unresolved. The first mismatch is not a proven globally minimal counterexample, and the observed error bound has not become a general tolerance.

## Successor: logical K prefixes and a two-term witness

Source `0cddd51` adds an explicit `prefix-control` suite with two logical families: `singleton` has M=N=1, and `dense` has M=N=16. Each scans every integer K from 0 through 16. The kernel bodies, physical 16×16 tile, full 64-thread WMMA participation, 256-thread scalar control, input formulas, exact oracle and three input snapshots remain unchanged. K=0 stores the zero accumulator; every positive K in this scan performs one source-level WMMA step with unused positions padded by zero. The [probe guide](../experiments/wmma/README.md#explicit-logical-prefix-suite) specifies its separate admission and metadata contract.

The plan fixed two complete 34-case sweeps before execution: forward cases with WMMA first, then reversed cases with scalar first in another process. It also preselected singleton K16 and dense K16 for separate WMMA-first traces. The traces were not selected from the measured error curve and are excluded from the earliest-K calculation.

All **70 scalar output matrices are exact**. WMMA has 20 exact matrices and 50 failing matrices, containing **931 unequal elements**: 450 in each complete sweep, one in the singleton trace, and 30 in the dense trace. All outputs are finite, all guards remain intact, and all **430,080 input-snapshot halfwords** match the prepared inputs and packing contract. All singleton padding outputs remain zero. These primary runs retain 140 output matrices and 1,400 timing batches; performance remains unaccepted.

The two sweep orders produce identical complete output words, including guards, for every corresponding case and implementation. At every K, both families also produce identical C[0,0] words for each implementation. The A row0 and B column0 words are identical across the families, including all three device snapshots. Changing the surrounding logical matrix from singleton to dense therefore did not change the observed C[0,0] response in these scans. This observation does not establish general independence from surrounding data.

The complete WMMA mismatch counts below are the same in both orders:

| K | Singleton: unequal elements | Dense: unequal elements | C[0,0] exact in both families? |
| ---: | ---: | ---: | --- |
| 0 | 0 | 0 | Yes |
| 1 | 0 | 0 | Yes |
| 2 | 0 | 1 | Yes |
| 3 | 0 | 2 | Yes |
| 4 | 1 | 42 | No |
| 5 | 1 | 99 | No |
| 6 | 0 | 47 | Yes |
| 7 | 0 | 22 | Yes |
| 8 | 1 | 22 | No |
| 9 | 1 | 31 | No |
| 10 | 1 | 33 | No |
| 11 | 0 | 21 | Yes |
| 12 | 0 | 12 | Yes |
| 13 | 1 | 17 | No |
| 14 | 1 | 33 | No |
| 15 | 1 | 29 | No |
| 16 | 1 | 30 | No |

For this complete declared grid, the earliest **any-output** residual is K=2 for dense and K=4 for singleton. The earliest C[0,0] residual is K=4 in both families. This is not a persistent threshold: C[0,0] becomes exact again at K=6,7,11,12. The minima apply only to these fixed input families; they are not globally minimal counterexamples or an admission rule for other inputs.

| Witness | Exact reference | Observed WMMA value | Reference / observed FP32 words |
| --- | ---: | ---: | --- |
| Dense K2, C[13,2] | -0.00390625 | -0.003906250465661287 | `0xbb800000` / `0xbb800001` |
| Singleton K4, C[0,0] | 0.0390625 | 0.039062488824129105 | `0x3d200000` / `0x3d1ffffd` |

The dense K2 witness is its matrix's only mismatch. Its two nonzero operand pairs are A=[-3/4, 1/16] and B=[-1/16, -13/16], giving products [3/64, -13/256] and exact sum **-1/256**. Scalar source returns that exact value; WMMA differs by `-2^-31`. The singleton K4 residual is `-3*2^-28`. These are arithmetic witnesses inside the declared physical tile, not measurements of an isolated native instruction. The largest absolute residual across the primary prefix cases is `5.960464477539063e-8`; it is not a general error bound.

Both preselected K16 traces match the complete outputs of their corresponding sweeps. Each contains 220 actual kernel events, split into separate 110-event WMMA/scalar groups with their own ten warmups. The tool reports WMMA block64/28 registers and scalar block256/36 registers, zero shared/private memory, and 110 false recompilation flags per group. Trace time units remain unverified, and neither timings nor resource descriptors establish the arithmetic cause.

The [complete prefix result](../data/results/20261008-wmma-prefix.json) also retains 12 old default cases and 12 old paired-control cases as regressions. Their 36 output buffers match the corresponding earlier public results. Across primary and regression runs, all 82 scalar matrices are exact; the overall record remains `correctness.passed=false` and `performance_accepted=false`. It contains 94 logical cases, 176 output matrices, 45,056 payload words, 22,528 guards, 192,512 prepared input halfwords, 503,808 captured input halfwords and 1,760 raw timing batches. All six workers and two profiled applications exited and passed release checks. No result was promoted to open-cake-ir.

The successor below tests the proposed isolation and relocation under a new input contract. The prefix scan itself does not perform that intervention or uniquely assign the residual to hardware, a compiler transformation, fragment loading or internal arithmetic.

## Successor: isolation and relocation of the two products

Source `6fd66b9` adds an explicit `witness-control` suite. All three patterns have the same logical shape **M=N=16, K=2**, physical 16×16 tile and launch geometry. Only the host-prepared operands and their independent reference change. Both device kernel bodies and launch helpers remain unchanged; the pattern selects no device branch. This source continuity does not assert binary identity with an earlier build. The [probe guide](../experiments/wmma/README.md#explicit-two-term-witness-suite) records the closed pattern/target/input-rule contract.

`dense-origin` retains the prefix study's formulas. `isolated-origin` keeps only A row13 and B column2, with ordered numerator pairs **[-12,1]** and **[-1,-13]**, each divided by 16. `isolated-c00` moves those same ordered pairs to A row0 and B column0. Every other operand position is positive zero, including the unused K positions and packed chunks. The isolated reference has one nonzero output, `-1/256`, at its declared target.

The plan fixed a forward WMMA-first sweep, a reversed scalar-first sweep in another process, and one separate WMMA-first trace for each pattern. Each pattern therefore has three retained paired observations, across five primary processes in total. The following target words are the same in all three observations:

| Pattern | Nonzero A / B halfwords | Target | WMMA FP32 word | Scalar / reference FP32 word |
| --- | ---: | --- | --- | --- |
| `dense-origin` | 31 / 31 | C[13,2] | `0xbb800001` | `0xbb800000` |
| `isolated-origin` | 2 / 2 | C[13,2] | `0xbb800001` | `0xbb800000` |
| `isolated-c00` | 2 / 2 | C[0,0] | `0xbb800001` | `0xbb800000` |

All **nine WMMA matrices have exactly one unequal element**, at the target. All **nine scalar matrices are exact**. The WMMA value is `-0.003906250465661287`; the reference is `-0.00390625`, a signed residual of `-2^-31`. Every other isolated output is numerically zero in both implementations. All values are finite, all guards intact, and all **55,296 primary input-snapshot halfwords** match the prepared inputs and fixed pattern contract.

Isolation masks 30 unrelated logical slots per operand, of which 29 were nonzero and actually change. Relocation changes four halfwords per operand: two original slots become zero and two destination slots receive the same words. Both ordered pairs remain bitwise identical through these interventions, including all three device readbacks. Complete output buffers match across the two sweep orders, and each trace matches both corresponding sweep outputs.

The prefix study's singleton C00/K2 reference is `153/256`. This relocated case preserves the original row13/column2 pairs and reference `-1/256`; it is a separate input contract. Neither K nor output position alone identifies the tested computation.

The residual therefore persists after clearing the dense surrounding entries and at both tested output coordinates. Those entries and the original coordinate are not necessary for this observed witness. This is a sparse two-product example inside a complete WMMA tile, not an isolated native-instruction measurement or evidence that all positions behave alike. Loading, lowering and internal arithmetic remain possible causes; no unique hardware attribution or general error bound follows.

Each of the three traces contains 220 actual kernel events, with separate 110-event WMMA/scalar groups and ten warmups per group. The tool reports WMMA block64/28 registers and scalar block256/36 registers; shared/private memory is zero and each group has 110 false recompilation flags. Exported time units remain unverified. Timings are retained for traceability and are not accepted as performance evidence.

The [complete witness record](../data/results/20261008-wmma-witness.json) retains all prepared inputs, snapshots and output words. Its primary scope is nine paired cases, 18 matrices, 4,608 payload values, 2,304 guard words and 180 timing batches. Separate default12, old-control12 and prefix34 regressions add 58 logical cases and 104 matrices; these auxiliary buffers match their corresponding previous results. Across all runs: 67 logical cases, 122 matrices, 31,232 payload values, 15,616 guards, 137,216 prepared halfwords, 337,920 captured halfwords and 1,220 raw batches. All 55 scalar matrices are exact; WMMA has 14 exact matrices out of 67, with 813 unequal elements in the others. Overall `correctness.passed=false` and `performance_accepted=false` remain explicit.

Frozen-source verification passed 136 CPU checks, all four native build modes, 15 input/format negatives and three incompatible-mode negatives. All eight device workers and three profiled applications exited and passed release checks. No result was promoted to open-cake-ir.

The successor below measures the two products separately and together under both K-slot assignments. The isolation/relocation study itself does not establish those component outcomes or a globally minimal counterexample.

## Successor: single products and K-slot permutations

Source `7b8ee81` adds a separate `product-control` contract. It holds **M=N=16, K=2 and target C[0,0]** fixed for all six conditions, including the one-product cases. The physical tile, full participation, both kernel bodies, launch helpers, scalar control and exact comparison policy remain unchanged. Each positive K case still executes one full source-level K16 WMMA step; reducing the number of nonzero products does not shrink the tile or loop.

The two nonzero operand pairs are inherited unchanged: `(-12/16)*(-1/16)=12/256` and `(1/16)*(-13/16)=-13/256`. A condition places a pair at K0 or K1, or places both pairs in the two available orders. Other input words remain positive zero. The [probe guide](../experiments/wmma/README.md#explicit-individual-product-and-k-slot-suite) records the complete vectors, admission rules and metadata. Probe flag `C550_WMMA_WITNESS=2` selects this new contract; mode 1 remains restricted to the earlier three-pattern witness suite. This is a deliberate probe-interface extension, not an SDK mode or a change to an earlier frozen record.

The plan fixed one six-condition WMMA-first sweep, its reverse with scalar first in a second process, and six separate preselected WMMA-first traces. Each condition has three paired observations, across eight primary processes. All three observations give these C00 words:

| Pattern | Product numerators at K0, K1 | Exact sum | WMMA word | Scalar / reference word |
| --- | --- | ---: | --- | --- |
| `positive-k0` | `[12, 0]` | 3/64 | `0x3d400000` | `0x3d400000` |
| `positive-k1` | `[0, 12]` | 3/64 | `0x3d400000` | `0x3d400000` |
| `negative-k0` | `[-13, 0]` | -13/256 | `0xbd500000` | `0xbd500000` |
| `negative-k1` | `[0, -13]` | -13/256 | `0xbd500000` | `0xbd500000` |
| `pair-forward` | `[12, -13]` | -1/256 | `0xbb800001` | `0xbb800000` |
| `pair-reversed` | `[-13, 12]` | -1/256 | `0xbb800001` | `0xbb800000` |

The four standalone-product conditions are exact in all **12 WMMA observations**. Both paired conditions fail in all **six WMMA observations**, each with only C00 unequal. All **18 scalar matrices are exact**. Every other primary output is numerically zero, all values are finite, guards are intact, and all **110,592 primary snapshot halfwords** match the prepared inputs and declared packing.

The separately observed component values are 0.046875 and -0.05078125. Their exact rational sum, computed on the CPU for this comparison, is `-1/256`. The joint WMMA result is `-0.003906250465661287`, differing by `-2^-31`. This CPU-derived relation is not a separately measured GPU addition kernel. It shows that the residual occurs in these joint-condition observations even though the corresponding one-product outputs are exact.

Complete buffers match when either standalone product moves between K0 and K1, and when the two nonzero pairs exchange K slots. They also match across the two execution orders and between each trace and its corresponding sweeps. These observations cover the stated values and two slots. A K-slot permutation is an input-placement intervention; it does not reveal the hardware's internal accumulation order or establish general associativity, position independence, or one-product exactness.

Each of the six traces contains 220 actual kernel events: separate 110-event WMMA/scalar groups, each with ten warmups. Resource descriptors remain WMMA block64/28 registers and scalar block256/36 registers, zero shared/private memory, and 110 false recompilation flags per group with none missing. Raw trace units remain unverified. Neither these descriptors nor the retained timings identify the arithmetic cause or support a performance comparison.

The [complete product record](../data/results/20261008-wmma-products.json) retains 18 paired primary cases, 36 output matrices, 9,216 payload values, 4,608 guards, 36,864 prepared halfwords and 360 timing batches. Auxiliary default12, old-control12, prefix34 and witness3 regressions add 61 logical cases and 110 output matrices; their full buffers match the corresponding prior records. Only the new `pair-forward` condition is compared to the prior `isolated-c00` witness as a matching complete input; its six variant outputs also match. Other product conditions are distinct inputs, not historical regressions.

Overall coverage is 79 logical cases, 146 matrices, 37,376 payload values, 18,688 guard words, 161,792 prepared halfwords, 411,648 captured halfwords and 1,460 raw batches. Scalar is exact in all 67 matrices. WMMA has 26 exact matrices out of 79, with 813 unequal elements in the remaining matrices. The aggregate remains `correctness.passed=false` and `performance_accepted=false`; passing standalone conditions do not change the failed paired conditions or authorize performance claims.

Frozen-source validation passed 145 CPU checks, five native build modes, 18 input/format negatives and three incompatible-mode negatives. All 12 device workers and six profiled applications exited and passed release checks. No result was promoted to open-cake-ir.

The successor below completes the proposed four-sign comparison with matching standalone controls; the current `(+12,-13)` condition supplies one existing baseline. A separately qualified SDK/compiler comparison would still need its own environment and lifecycle evidence. The current result applies to the recorded MACA 3.5.3.18 route.

## Successor: fixed-magnitude sign configurations

Source `d1e1be9` adds a separate `sign-control` contract with eight fixed conditions. All hold M=N=16, K=2, target C[0,0], full physical participation and the exact comparison policy. The device kernels and launch helpers remain unchanged. Mode 3 is an explicit probe selector; older modes retain their own input contracts. The [probe guide](../experiments/wmma/README.md#explicit-product-sign-suite) specifies all vectors and metadata.

The four joint cases keep B numerators **[-1,-13]** fixed and change only A signs: A is [-12,-1], [-12,1], [12,-1] or [12,1]. Operands are divided by 16. Four matched standalone controls test ±12/256 at K0 and ±13/256 at K1, setting both operands of the inactive term to positive zero. All unrelated input words remain positive zero. Transferring a sign between A and B while preserving a mathematical product changes this declared factorization and is refused under an unchanged label.

The plan fixed a forward eight-case WMMA-first sweep, a reversed scalar-first sweep in another process, and one separate WMMA-first trace per condition. Each condition has three paired observations, across ten primary processes. All four standalone controls are exact. The joint outcomes are identical across their three observations:

| Joint pattern | Product numerators at K0, K1 | Exact C00 | WMMA word | Scalar / reference word | Signed residual |
| --- | --- | ---: | --- | --- | ---: |
| `pair-pp` | `[12,13]` | 25/256 | `0x3dc80000` | `0x3dc80000` | 0 |
| `pair-pn` | `[12,-13]` | -1/256 | `0xbb800001` | `0xbb800000` | -2^-31 |
| `pair-np` | `[-12,13]` | 1/256 | `0x3b7ffffe` | `0x3b800000` | -2^-31 |
| `pair-nn` | `[-12,-13]` | -25/256 | `0xbdc80000` | `0xbdc80000` | 0 |

The mixed-sign target values are `-0.003906250465661287` and `0.0039062495343387127`. **Both signed residuals are negative**, despite opposite reference sums. Negating the two nonzero A terms does not exactly negate the observed target in this pair of cases. Their measured values sum to `-2^-30` under exact CPU arithmetic, while the references sum to zero. This is a derived relation between retained outputs, not a new GPU addition or negation experiment.

The representable-value distances differ: `pair-pn` is one adjacent FP32 step from its reference, while `pair-np` is two. The positive reference lies on an exponent boundary, and the spacing below it is smaller. Equal absolute error `2^-31` must not be labeled one ULP for both without defining the convention. These two examples do not establish a general rounding mode or a universal negative bias.

All **24 scalar matrices are exact**. WMMA is exact in 18 matrices: all standalone controls and both same-sign joint conditions. The six mixed-sign observations each have one unequal element at C00. Every other primary output is numerically zero, all values are finite, guards are intact, and all **147,456 primary input-snapshot halfwords** match their prepared and declared values. Full buffers agree across execution orders and between each trace and the matching sweeps.

This comparison observes a difference between the stated sign configurations under one factorization. It also changes the exact result magnitude from 25/256 to 1/256. It does not uniquely separate cancellation severity, result magnitude, loading, lowering or internal arithmetic, and it does not show that opposite-sign products generally fail. A different placement of sign bits between operands is an untested input treatment.

The eight traces each contain 220 actual kernel events, split into 110-event WMMA/scalar groups with ten warmups apiece. Descriptors remain WMMA block64/28 registers and scalar block256/36 registers, zero shared/private memory, and 110 false recompilation flags per group with no missing values. Trace units remain unverified. Timing is retained without performance acceptance or an instruction-level interpretation.

The [complete sign record](../data/results/20261008-wmma-signs.json) contains 24 paired primary cases, 48 matrices, 12,288 payload values, 6,144 guards, 49,152 prepared halfwords and 480 timing batches. Five auxiliary suites add 67 logical cases and 122 matrices, all matching their corresponding prior buffers. Only `positive12-k0`, `negative13-k1` and `pair-pn` match earlier product inputs; their 18 variant observations also match those prior outputs. The other sign conditions are distinct input contracts.

Overall coverage is 91 logical cases, 170 matrices, 43,520 payload values, 21,760 guards, 186,368 prepared halfwords, 485,376 captured halfwords and 1,700 raw batches. Scalar is exact in all 79 matrices; WMMA has 36 exact matrices out of 91, with 815 unequal elements in the remainder. The aggregate remains `correctness.passed=false` and `performance_accepted=false`. All 15 workers and eight profiled applications exited and passed release checks. Frozen source passed 154 CPU checks, six native builds, 21 input/format negatives and three incompatible-mode negatives. No result was promoted to open-cake-ir.

The successor below extends the proposed neighboring-magnitude comparison to every integer q from 1 through 14, retaining the exact sum -1/256 and matched standalone controls. The sign study itself does not establish those outcomes. Product-preserving sign transfers and separately qualified SDK/compiler comparisons remain open questions.

## Successor: adjacent product magnitudes at a fixed result

Source `9b6c28c` adds a separate `magnitude-control` contract with **42 conditions: q=1…14, each with positive, negative and paired roles**. Every condition fixes M=N=16, K=2, target C[0,0], the full physical tile, exact comparison policy and existing kernel/launch bodies. Probe mode 4 selects this closed input table; older modes retain their contracts. The [probe guide](../experiments/wmma/README.md#explicit-adjacent-magnitude-suite) defines the exact inputs and admission rules.

For each q, the active vectors and products are:

| Role | A numerators at K0, K1 | B numerators at K0, K1 | Exact C00 |
| --- | --- | --- | ---: |
| Positive component | `[-q, 0]` | `[-1, 0]` | q/256 |
| Negative component | `[0, 1]` | `[0, -(q+1)]` | -(q+1)/256 |
| Pair | `[-q, 1]` | `[-1, -(q+1)]` | -1/256 |

All operands are divided by 16. Every other input word is positive zero; every other reference output is zero. The q range keeps input numerators inside the existing encoder's ±15 domain. It is an experiment bound, not a hardware limit.

The plan fixed a complete 42-condition WMMA-first sweep, the reversed 42 conditions with scalar first in another process, and three separate preselected pair traces at q1, q12 and q14. **Both standalone components are exact at every q in both sweeps. All scalar matrices are exact.** Complete output buffers, including guards, agree across the two sweeps. Paired WMMA results are:

| q | Product numerators on the /256 scale | Paired WMMA C00 in both sweeps | Signed residual |
| ---: | --- | --- | ---: |
| 1 | `[1, -2]` | `0xbb800000` | 0 |
| 2 | `[2, -3]` | `0xbb800000` | 0 |
| 3 | `[3, -4]` | `0xbb800000` | 0 |
| 4 | `[4, -5]` | `0xbb800000` | 0 |
| 5 | `[5, -6]` | `0xbb800000` | 0 |
| 6 | `[6, -7]` | `0xbb800000` | 0 |
| 7 | `[7, -8]` | `0xbb800001` | -2^-31 |
| 8 | `[8, -9]` | `0xbb800001` | -2^-31 |
| 9 | `[9, -10]` | `0xbb800001` | -2^-31 |
| 10 | `[10, -11]` | `0xbb800001` | -2^-31 |
| 11 | `[11, -12]` | `0xbb800001` | -2^-31 |
| 12 | `[12, -13]` | `0xbb800001` | -2^-31 |
| 13 | `[13, -14]` | `0xbb800001` | -2^-31 |
| 14 | `[14, -15]` | `0xbb800001` | -2^-31 |

The paired reference is always `0xbb800000`, or -0.00390625. The failing value is -0.003906250465661287, one adjacent FP32 step below it. Each failing matrix has exactly one unequal element, at C00. Every other primary output is numerically zero, all outputs are finite, guards are intact, and all **534,528 primary snapshot halfwords** match prepared inputs and the fixed contract. The three traced outputs match their corresponding sweep buffers: q1 is exact, while q12 and q14 retain the residual.

Thus q7 is the first observed paired mismatch **within this declared integer grid and factorization**. Equal final reference values do not identify equal numerical behavior. Increasing q also changes product magnitudes, relative cancellation and operand encodings together; this comparison does not isolate their contributions. It establishes neither a global threshold nor behavior outside this grid, and it does not identify internal precision, rounding, loading or compiler behavior as the cause.

Only the q1, q12 and q14 pairs have primary profiler evidence. No standalone condition or q6/q7 boundary condition was profiled. Each of the three traces contains 220 actual kernel events, split into 110-event WMMA/scalar groups with ten warmups each. Descriptors remain WMMA block64/28 registers and scalar block256/36 registers, zero shared/private memory, and 110 false recompilation flags per group. Raw trace units remain unverified; timings have no performance acceptance.

The [complete magnitude record](../data/results/20261008-wmma-magnitudes.json) retains all input, snapshot and output words. Primary coverage is **87 paired logical cases, 174 matrices and 1,740 timing batches**. WMMA is exact in 69/87 matrices; the other 18 each have one C00 mismatch. Scalar is exact in 87/87. Six auxiliary suites add 75 logical cases and 138 matrices, all matching earlier complete buffers. Only the q12 trio matches the prior sign-study inputs: its component conditions each have two sweep observations, and its pair has two sweeps plus a trace. These contribute 14 historical variant comparisons; with the auxiliary comparisons, all 152 agree.

Overall coverage is 162 logical cases, 312 matrices, 79,872 payload words, 39,936 guards, 331,776 prepared halfwords, 921,600 snapshot halfwords and 3,120 raw batches. All 150 scalar matrices are exact; WMMA has 93/162 exact matrices and 829 unequal elements in the remainder. The aggregate remains `correctness.passed=false` and `performance_accepted=false`. Frozen source passed 163 CPU checks, seven native builds, 26 input/format negatives and three incompatible-mode negatives. All 11 device workers and three profiled applications exited and passed release checks. No result was promoted to open-cake-ir.

The successor below performs the proposed A-scaling comparison at q6, q7 and q12 under a new exact dyadic input contract. The magnitude study itself does not establish those scale outcomes. Product-preserving factor/sign placement and independently qualified SDK comparisons remain separate open questions.

## Successor: exact power-of-two scaling of A

Source `2725811` adds a separate `scale-control` contract: q in **{6,7,12}**, exponent e in **{-2,-1,0,1,2}**, and positive, negative or paired roles. All conditions keep M=N=16, K=2, target C[0,0], the full physical tile and the existing device/launch bodies. Only host-prepared operands change. All new conditions use the same mode-5 binary; equality of source bodies does not assert binary identity with earlier builds. The [probe guide](../experiments/wmma/README.md#explicit-exact-dyadic-a-scale-suite) specifies the dyadic encoding, metadata and admission rules.

For the pair, **A=2^e×[-q,1]/16** and **B=[-1,-(q+1)]/16**. Positive and negative controls retain their corresponding term and set both operands of the inactive term to positive zero. The paired reference is exactly **-2^e/256**. Scaling A preserves the two products' ratio and relative cancellation while changing absolute operand, product and result magnitudes. Every nonzero input and target reference is exactly representable as a normal binary16 value. All other input words are positive zero, and all other reference outputs are zero.

The plan contains **45 parameter conditions and 41 distinct complete A/B pairs**. Four positive-control aliases connect q6 at e=-1,0,1,2 to q12 at e=-2,-1,0,1. These planned conditions remain separately recorded; they are not counted as distinct input data. Both complete sweeps use opposite case and implementation orders. Three preselected traces cover only the q7 pair at e=-2,0,+2, giving **93 paired primary observations** in total.

Every standalone component is exact in both WMMA sweeps. All scalar matrices are exact. The paired target words agree across both sweeps:

| e | Exact C00 and q6 WMMA word | q7 and q12 WMMA word | q7 and q12 signed residual |
| ---: | --- | --- | ---: |
| -2 | -1/1024 · `0xba800000` | `0xba800001` | -2^-33 |
| -1 | -1/512 · `0xbb000000` | `0xbb000001` | -2^-32 |
| 0 | -1/256 · `0xbb800000` | `0xbb800001` | -2^-31 |
| 1 | -1/128 · `0xbc000000` | `0xbc000001` | -2^-30 |
| 2 | -1/64 · `0xbc800000` | `0xbc800001` | -2^-29 |

Each failing matrix has exactly one unequal element at C00. For q7 and q12, the signed residual is **r=-2^(e-31)** at every tested exponent, so **r/2^e=-2^-31** throughout this grid. The absolute error changes with scale; each result remains one adjacent FP32 value below its own reference. q6 has zero residual throughout. These three metrics describe different quantities and must remain separate.

The exact CPU sum of the two observed standalone targets equals the paired reference in each condition. Subtracting that sum from the observed paired WMMA target gives the residual above. This is derived arithmetic on retained outputs, not an additional GPU addition measurement. The finite observations show a consistent normalized residual for these failing conditions; they establish neither a general scaling law nor an internal precision, rounding mode or unique hardware/compiler cause. No new full q scan was performed at each scale.

Complete buffers agree across the two orders, between each trace and its corresponding sweeps, and within the four full-input alias pairs. All other 255 primary outputs are numerically zero, all outputs are finite, guards remain intact, and all **571,392 primary snapshot halfwords** match prepared and declared inputs. Of the 93 primary WMMA matrices, **70 are exact and 23 have one C00 mismatch**. All **93 scalar matrices are exact**.

Only q7 at e=-2,0,+2 has primary profiler evidence; no component, q6, q12 or e=±1 condition was profiled. Each trace has 220 actual kernel events, split into 110-event WMMA/scalar groups with ten warmups apiece. Descriptors remain WMMA block64/28 registers and scalar block256/36 registers, zero shared/private memory, and 110 false recompilation flags per group. Trace units remain unverified, and no timing is accepted as a performance result.

The [complete scale record](../data/results/20261008-wmma-scales.json) retains all actual words and raw timing batches. Primary coverage is 93 logical cases, 186 matrices, 47,616 payload words, 23,808 guards, 190,464 prepared halfwords and 1,860 batches. Seven auxiliary suites add 117 logical cases and 222 matrices, all matching their corresponding prior buffers. Fourteen primary conditions match earlier magnitude inputs by complete operand equality, including five matches at nonzero e. Their 58 variant observations also agree with the earlier outputs. Thus all **280 historical comparisons** agree; matching only shape or e=0 would miss part of this coverage.

Overall coverage is 210 logical cases, 408 matrices, 104,448 payload words, 52,224 guards, 430,080 prepared halfwords, 1,216,512 snapshot halfwords and 4,080 raw batches. Scalar is exact in 198/198 matrices. WMMA has 128/210 exact matrices and 842 unequal elements in the remainder. The aggregate remains `correctness.passed=false` and `performance_accepted=false`.

Frozen source passed 172 CPU tests. A separate host-only C++ check verified 155 encoder inputs and four domain refusals; it is not GPU coverage. All eight native modes compiled, and 31 input/format negatives plus three incompatible-mode negatives passed with devices hidden. The SSH connection carrying the original CPU build sequence closed with exit 255; subsequent inspection found five completed builds, no remaining compiler process and three untouched modes. A separately retained continuation built only those three modes. The SSH exit was 255; the original remote driver's exit and the disconnect's cause remain unknown. No build record was overwritten. All **12 device workers and three profiled applications** subsequently exited and passed release checks. No result was promoted to open-cake-ir.

The successor below tests reciprocal scaling under a separate input contract, preserving the exact products and reference while redistributing operand exponents. The A-only study itself does not establish those outcomes. Product-preserving sign placement and independently qualified SDK comparisons remain open questions.

## Successor: reciprocal exponents with fixed products

Source `b1eefc4` adds a separate `reciprocal-control` contract with **45 conditions and 45 distinct complete A/B pairs**. It tests q in {6,7,12}, A exponent e in {-2,-1,0,1,2}, and positive, negative or paired roles. Logical M=N=16, K=2, target C[0,0], physical participation, both device kernels and launch helpers remain unchanged. The [probe guide](../experiments/wmma/README.md#explicit-reciprocal-input-scale-suite) defines the new mode-6 header and the two explicitly bound exponents.

For the pair, **A=2^e×[-q,1]/16** and **B=2^-e×[-1,-(q+1)]/16**. Each exact product remains q/256 or -(q+1)/256, and the paired reference stays **-1/256**. Matching component controls set both operands of the inactive term to positive zero. All other input words are positive zero and all other reference outputs are zero. The nonzero operands remain exact normal binary16 values, including the largest magnitude 13/4 in B. The existing dyadic encoders are reused without widening their domain.

The plan fixed two complete 45-condition sweeps in opposite case/implementation orders, plus three preselected q7 pair traces at e=-2,0,+2. All primary conditions use the same compiled mode-6 binary. This source continuity does not assert binary identity with an earlier study.

Every standalone component and every scalar matrix is exact. The paired WMMA C00 words are identical in both complete sweeps:

| A exponent e | B exponent -e | q6 WMMA | q7 WMMA | q12 WMMA |
| ---: | ---: | --- | --- | --- |
| -2 | 2 | `0xbb800000` | `0xbb800001` | `0xbb800001` |
| -1 | 1 | `0xbb800000` | `0xbb800001` | `0xbb800001` |
| 0 | 0 | `0xbb800000` | `0xbb800001` | `0xbb800001` |
| 1 | -1 | `0xbb800000` | `0xbb800001` | `0xbb800001` |
| 2 | -2 | `0xbb800000` | `0xbb800001` | `0xbb800001` |

The paired reference is always `0xbb800000`, or -0.00390625. q6 is exact throughout. Each q7/q12 paired matrix has only C00 unequal, with observed value -0.003906250465661287: signed residual **-2^-31**, absolute error **2^-31**, and **one adjacent FP32 step below the reference**. These residuals are measured against a fixed reference; dividing them by A's scale would not describe whole-output scaling in this experiment.

For every fixed q, role, implementation and sweep order, the complete output buffers are identical across the five reciprocal exponent pairs, including guards. Corresponding buffers also match across the two execution orders and between each trace and both sweeps. Every other primary output is numerically zero, all values are finite, guards are intact, and all **571,392 primary snapshot halfwords** match their prepared and declared values.

The exact CPU sum of the observed standalone targets equals the paired reference in each condition. Subtracting that sum from the observed paired WMMA target gives the signed residual above. This is derived arithmetic on retained outputs, not a separate GPU addition test. The prior A-only study changed the product and result scale. This study holds those quantities fixed and observes no output change under the five tested exponent redistributions. It does not establish invariance to arbitrary factorization, sign placement, subnormal inputs or other software versions, and it does not identify the internal arithmetic or a unique hardware/compiler cause.

Only the q7 pair at e=-2,0,+2 has primary profiler evidence. No component, q6, q12 or e=±1 condition was profiled. Each trace contains 220 actual kernel events, with separate 110-event WMMA/scalar groups and ten warmups each. Descriptors remain WMMA block64/28 registers and scalar block256/36 registers, zero shared/private memory, and 110 false recompilation flags per group. Raw trace units remain unverified. No timing is accepted as a performance result.

The [complete reciprocal record](../data/results/20261008-wmma-reciprocal.json) retains all actual words and raw batches. Primary coverage is **93 paired logical observations, 186 matrices and 1,860 batches**, with 47,616 payload words, 23,808 guards and 190,464 prepared halfwords. WMMA is exact in 70/93 matrices, with 23 unequal elements in the other 23 matrices. Scalar is exact in 93/93. Eight auxiliary suites add 162 logical cases and 312 matrices, all matching the corresponding earlier buffers.

Historical matching examines every complete prior A/B array and retains all matches. Nine primary conditions match earlier one-sided-scale data, forming eleven condition-to-prior-pattern links because two old positive controls have aliases. Across the two sweeps and the matching trace, **38 currently measured primary variant outputs produce 46 historical comparison rows**. Adding the 312 auxiliary rows gives **358 comparisons covering 350 current variant outputs**; all agree. Comparisons to multiple old aliases are not additional device measurements.

Overall coverage is 255 logical cases, 498 matrices, 127,488 payload words, 63,744 guards, 522,240 prepared halfwords, 1,492,992 snapshot halfwords and 4,980 raw batches. Scalar is exact in 243/243 matrices. WMMA has 163/255 exact matrices and 852 unequal elements in the remainder. The aggregate remains `correctness.passed=false` and `performance_accepted=false`.

Frozen source passed 181 CPU tests. A separate host-only compilation of the frozen metadata emitter matched all 15 protocol fields and all 45 pattern records, including plural historical lists. Nine native modes compiled through separate retained CPU stages; 38 input/format negatives and three incompatible-mode negatives passed with devices hidden. All **13 device workers and three profiled applications** exited and passed release checks. No result was promoted to open-cake-ir.

A proposed next contrast can transfer each nonzero term's sign between A and B while preserving its signed product, with matching component controls. The current reciprocal study preserves sign placement, so it cannot establish that response. Such a contrast needs a separate input contract and has not been executed. Native codegen investigation and independently qualified SDK comparisons remain open.

[wmma]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#warp-matrix
[types]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#nhvxy67mk8uv1
