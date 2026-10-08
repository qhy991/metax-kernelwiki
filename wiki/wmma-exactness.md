# Native C550 WMMA: failed exact acceptance and reproducible residuals

[Home](../README.md) · [Catalog](../data/catalog.json) · [Probe guide](../experiments/wmma/README.md)

In this C550 / MACA 3.5.3.18 / MXCC `1.0.0 (6477545d4d)` environment, native 16×16×16 FP16 WMMA with a float accumulator fragment compiles and executes, but **does not satisfy the strict exact numerical contract for these inputs**. In the first 12 cases, K=0 and 1×1×1 pass exactly. The other 10 cases contain 177 outputs unequal to the predetermined reference. Prepared inputs are valid, guards are intact and all outputs are finite.

The first mismatch is C[0,0] for 16×16×16: reference `-0.5` (`0xbf000000`), observed `-0.5000000596046448` (`0xbf000001`). Two independent processes and a diagnostic trace using the same frozen binary reproduce the complete original output words. The [raw inputs, outputs and diagnostic record](../data/results/20261008-wmma-exact-diagnostic.json) explicitly retain `correctness.passed=false` and `performance_accepted=false`. No tolerance is relaxed and no performance conclusion is drawn from the timings.

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

A next study can reduce logical matrix extents and K prefixes while preserving the complete physical tile and scalar control, then examine when residuals appear. Actual generated code still needs inspection. The first mismatch is not a proven globally minimal counterexample, and the observed error bound has not become a general tolerance.

[wmma]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#warp-matrix
[types]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#nhvxy67mk8uv1
