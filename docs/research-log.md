# Research log

[Home](../README.md)

## 2026-10-07: Establishing the C550 evidence baseline

Research question: how can native MACA interfaces, compiler behavior, and small counterexample experiments produce useful knowledge for C550 kernel development?

Both reference repositories were inspected, and an index of official MACA documents and pinned mcTriton source was prepared. SSH access, the MACA SDK, and an existing test container were visible on C550-1. All eight GPUs held vLLM worker contexts using approximately 64 GiB per card. Zero utilization did not establish availability for exclusive use. Those workloads were left untouched.

Initial experimental questions:

1. Do the product name, ISA, native runtime properties, and compilation target consistently identify the selected C550?
2. Is copy elementwise correct across block sizes, including tails?
3. With a fixed output element count, how do read strides 1/2/4/8/16 affect event intervals and effective data rate?
4. How does empty-kernel event time change with batch size? Can fixed event overhead be distinguished from the interval between dispatches?

The planned progression follows evidence: copy/transpose → shared-memory padding and barriers → 64-lane reduction/shuffle boundaries → matrix compilation routes and profiling. Each round should propose only mechanisms that its data can distinguish. Without profiler or assembly evidence, do not infer bank count, cache-line geometry, or matrix-instruction throughput.

## Initial device results

`20261007-native-01`, using source `cbdea92` on one C550 with MACA 3.5.3.18, completed 49 cases: 45 with outputs and 4 empty kernels. All 22,042,413 valid outputs and 2,880 guard words passed. The pinned source checkout also passed 16 CPU negative-control/protocol tests. Complete CPU checking ran after the GPU process exited; release observations found no remaining MACA lock or GPU process belonging to the run.

[Full results and 490 raw event batches](../data/results/20261007-native-01.json) · [Interpretation](../wiki/memory-access.md).

In this fixed-order sweep, elementwise copy of approximately 16 MiB took about 67.7/57.6/47.5/43.4 μs per launch within event batches for blocks 64/128/256/512. The stride sweep changes both transaction access shape and working-set span, so its trend cannot isolate caching or coalescing. [Balanced-order confirmation in four independent processes](../data/results/20261007-copy-confirm.json) passed, with the same ranking in all four. The within-process `T64/T512` ratio was 1.528 [1.522, 1.532]. The analysis unit is a process; 40 batches are not 40 independent replications.

## A profiler failure and its successor capture

`20261007-trace-01` showed that mcTracer 3.5.3.18 appends an absolute `--odname` to its working directory, fails to create the resulting path, yet exits with code 0. That failure is retained. `20261007-trace-02` used relative `--odname trace` in a fresh directory, produced a trace, and passed full-output checking for that copy run. [Trace content checks](../wiki/profiling.md) established 1020 actual GPU kernel events paired one-to-one with host launches. Exported time units remain independently unverified: raw values are retained without conversion to calibrated microseconds. File existence or exit code alone is insufficient.

## 2026-10-08: 512/1024 function attributes and recompilation paths

Device execution and tracing completed on 2026-10-07 and were summarized the next day. Successor source `088783d` passed an 18-case boundary check: a 1024-thread block executed correctly while the default function's `mcFuncGetAttributes.maxThreadsPerBlock` still returned 512. A single-case trace reported the default 1024-thread copy's execution variant as `max_block_size=1024`; `is_recompiled=true` appeared in 949 events, while the field was absent from another 72.

Two builds from commit `bc9f748` then compared the default declaration with explicit `__launch_bounds__(1024)`, holding the copy body, block=1024, inputs, oracle, and other compilation options fixed. The explicit version passed all 18 boundary cases. Its separate trace reported function attribute 1024, false recompilation flags for all 949 events that supplied the field, and no files in the fresh binary-cache directory. The default control still reported 512, true recompilation flags, and two cache files. First-launch host completion intervals were 164.734/2.659 ms, with only one profiled observation per condition; no speedup ratio is claimed.

[Mechanism page](../wiki/launch-bounds.md) · [Default boundaries and raw trace samples](../data/results/20261007-block1024-boundary.json) · [Same-source explicit-bound control](../data/results/20261007-launch-bound-control.json).

The first request through the old user-level lock was refused because of cooperating work on other cards; the probe did not execute, and the failure is retained. The successor caller explicitly used the node's deployed device-level owner to obtain device 0 under the same lock protocol. It stopped no other task, deleted no lock, and added no allocator. All device processes above exited and passed release checks.

The official 3.5.3.x runtime and mcTracer manuals were rechecked. The initial `/preview/567` URL actually selected version 3.0.0.x; the current reference was corrected while preserving that history. No explicit mcTracer export-unit contract was found, so raw values remain unconverted.

These findings entered this wiki without changes or promotion to open-cake-ir's Compiler, Target, or calibration. The earlier `eb6021e` GitHub CPU workflow was confirmed successful. New control source `bc9f748` passed 39 CPU checks in a clean independent checkout.

## 2026-10-08: Read permutations over a fixed address set

The probe at `0e093ce` reads the complete `0..N-1` set for each N, writes outputs contiguously, and reuses the same input/output allocations within a process. Only the read permutation changes. An independent CPU matrix-transpose index checks the outputs. All 18 parameter cases passed complete checking, followed by 36 passing cases at N=2^24 in six predetermined balanced orders.

A pronounced nonmonotonic effect remained with total span fixed: within-process `T(s=6)/T(s=12)` was 3.284 [3.258, 3.288], with the same direction in all six processes. Larger adjacent read spacing cannot simply be interpreted as slower, and total input span alone cannot explain this difference. Three independent single-case traces each contained 110 kernels and reported 8 registers with zero shared/private memory, with no observed change in recompilation flags. No memory-traffic counters identify a unique cause.

[Mechanism page](../wiki/memory-order.md) · [570 event batches, all six processes, and three traces](../data/results/20261008-memory-order.json). One input set was checked repeatedly across parameters and processes; output-element checks are not a count of distinct random inputs. All ten device processes exited and passed allocation-release checks.

No result was promoted to open-cake-ir. Public-repository CI for `9411468` passed, and the pinned probe source passed 52 CPU checks in an independent checkout.

## 2026-10-08: Shared-memory tiling for the same transpose result

Source `bf1db0f` added direct, tile64, and tile64_pad1 implementations of the same row-major FP32 transpose. All 57 cases across 19 shapes passed, including thin one-dimensional shapes, 31×33/33×31, 63/64/65 boundaries, and two large ragged matrices. Each valid store has the same guard as its corresponding shared-memory loader, and the entire block reaches the barrier unconditionally.

All 54 confirmation cases across six processes and three large shapes also passed. The median within-process speedups of unpadded tile64 over this direct baseline were 9.183, 7.523, and 2.833. Padding had no uniform direction: approximately neutral for 262144×64, slightly slower for 65536×256, and slightly faster for 4096×4096. Operation count, block count, per-thread work, shared memory, and barriers change together, so the improvement cannot be assigned to one mechanism alone.

[Mechanism page](../wiki/transpose.md) · [1140 raw event batches and three independent traces](../data/results/20261008-transpose.json). Each trace contained 110 kernels. Reported register counts were 14 for direct and 13 for tiled kernels; shared usage was 0/16384/16640 bytes. No bank-conflict or DRAM counters were collected. Official C500 material served as prior information, not as established C550 bank constants.

All ten device processes and profiled applications exited, with no allocation retained for this task. This remained an independent native experiment, with no promotion to open-cake-ir's Compiler, Target, or calibration. Pinned source passed 63 CPU checks in an independent checkout; public CI for `1166d12` passed.

## 2026-10-08: Shared pitch with fixed capacity and one function

Source `56db27e` fixes 4160 shared elements in one non-template kernel and selects pitch 64/65 at runtime. The full 38-case check passed. Ten later processes covered five large shapes, with each shape appearing twice at every position, once in each pitch order; all 100 confirmation cases passed. Every process for each of the five shapes observed pitch65 as faster, with median within-process ratios ranging approximately 1.079×–1.189×.

Two traces each contained 110 kernels with the same function name and reported 13 registers and 16640 static shared bytes. The default 57-case suite and old record protocol were preserved. This control reduces the first round's capacity and template-instance confounding, but active shared addresses still change. It does not identify a bank mapping or uniquely attribute differences between old and new implementations to capacity or occupancy.

[Follow-up on the same mechanism page](../wiki/transpose.md#successor-one-function-with-a-fixed-16640-byte-capacity) · [Independent result and 1400 timing batches](../data/results/20261008-shared-pitch.json). All thirteen device processes and profiled applications exited. Nothing was promoted to open-cake-ir. The pinned source passed 66 CPU tests in an independent checkout; public CI for `820fc44` passed.

## 2026-10-08: Separating dynamic shared requests from pitch

Source `dd20525` added three configurations of one dynamic-shared kernel: pitch64 requesting 16,384/16,640 B, and pitch65 requesting 16,640 B. A 57-case sweep was followed by 200 confirmation cases in ten processes with predetermined A_first/B/C/A_last or A_first/C/B/A_last order, plus three separate traced cases. All 260 cases were correct, with 2600 raw batches retained.

The fixed-pitch A/B comparison consistently favored the smaller request in four of five shapes. The 4095×4097 ratio crossed 1, so no uniform capacity rule is claimed. At a fixed request, B/C favored pitch65 in all ten processes for all five shapes. Both repeated endpoints and each endpoint's separate ratio to B were retained; processes with greater drift were not filtered out.

[Follow-up control](../wiki/transpose.md#further-control-dynamic-shared-memory-requests-in-one-function) · [Complete result](../data/results/20261008-dynamic-shared.json). All three traces used the same function name, with 110 kernels each and reported 13 registers / zero static shared memory. Reported dynamic shared values were 16384/16640, matching the requests; all recompilation flags were false. Physical allocation granularity and actual occupancy remain unidentified.

All 14 device workers and three profiled applications were released. Pinned source passed 70 CPU checks in an independent checkout, local MXCC compilation, and host-side negative controls for illegal combinations. No result was promoted to open-cake-ir. Fixed-capacity pitch result `a61ea94` was published with the user's explicit authorization and passed public CI; equivalently validated increments continued to be published.

The official C++ guide 3.5.3.x shuffle, integer-reduction, and synchronization contracts were also checked, with the original page and active version selector retained. At that point, this repository had not yet measured those interfaces on C550. See the [toolchain page](toolchain.md#64-lane-collective-contracts-and-measured-scope) for the interface basis and subsequent scope.

## 2026-10-08: Shuffle and integer reduction on complete wave64 groups

Source `d01ef59` added 20 boundary cases: 64/128 physical threads, logical n across 31/32/33, 63/64/65, and second-wave boundaries such as 95/96/97, plus n0. Every physical thread executes eight direct-shuffle queries and a signed-int sum with the full 64-bit mask, saving all nine results. Logical tails contribute zero without early exit.

The forward 20-case sweep, an independent reverse-order 20-case process, and two single-case traces passed full checking: 39744 payload values, 2688 guards, and 420 raw event batches in total. With 128 complete threads, the two wave sums were 2080 and 6176. The upper width32 subgroup and second wave selected sources within their own groups; source32/63 modulo behavior agreed with the documentation and installed headers. At logical n65, zero-padded thread127 still received reduction value 65, showing why checking only the first n threads would omit meaningful outputs.

[New mechanism page](../wiki/wave-collectives.md) · [Results with every actual integer output](../data/results/20261008-wave-collectives.json). The two traces each contained 110 kernels, with blocks 64/128 respectively; both reported 16 registers, zero shared/private memory, and false recompilation flags. A source helper does not prove a single hardware instruction. Retained complete-kernel times are not intrinsic latency or throughput results.

Four device workers and two profiled applications exited and passed release checks. Clean independent source passed 83 CPU checks and local MXCC compilation. With devices hidden, block32/n32 and block64/n65 negative controls were refused on the host. Nothing was promoted to open-cake-ir. Public CI for the earlier `9a23cb8` release passed.

## 2026-10-08: SDK overload boundaries for complete typed masks

Source `715e877` retained the default nine-channel probe and added a two-channel mask-types mode. Compilation checked unsigned32, unsigned-long64, uint64_t identity, both literal types, and the SDK's `MACA_HALF_WARP_SIZE=32`. The two typed variables go directly to native overloads, without a common wide-mask wrapper or a truncated partial 64-bit-mask call.

All 20 forward cases, 20 reverse cases, and two traced cases in the new mode passed. With a complete 128-thread block, the 64-bit channel returned 2080/6176, while the 32-bit compatibility channel returned 528/1552/2576/3600. At n65/thread127 the outputs were [65,0]. These are different grouping contracts for two APIs. The physical wave remains 64, and both mask type and value differ; no type-only causal or speed ratio is claimed.

[Follow-up on the same mechanism page](../wiki/wave-collectives.md#successor-full-typed-masks-select-different-reduction-interfaces) · [All actual outputs and raw batches](../data/results/20261008-mask-overloads.json). A separate 20-case default mode0 regression also passed, bringing the total to 62 cases, 27840 payload values, 3968 guards, and 620 batches. Each new-mode trace contained 110 kernels, reporting 16 registers, zero shared/private memory, and false recompilation flags.

All five device workers and two profiled applications exited. Clean source passed 89 CPU checks, both local compilation modes, and four host negative controls with devices hidden. The first evidence collection omitted empty directories, so the strict projection refused it. A successor read-only archive preserved the original remote empty directories and passed without changing the experiment or manufacturing directory state. Nothing was promoted to open-cake-ir. Public CI for the earlier `1f6a50a` release passed.

## 2026-10-08: A native WMMA exactness counterexample

The absence of torch/triton from system Python did not mean the machine lacked a framework environment. Read-only package metadata inspection of an existing containerd container found Torch 2.10.0, Triton 3.6.0, and maca-tile 1.0.1 with suffix `metax3.8.0.4.c600u`. No framework was imported or its GPU path executed. That environment is recorded separately from the host MACA 3.5.3 experiment and inherits no execution qualification.

Source `ef22b51` used the installed SDK's native `mxmaca::wmma` header for FP16 inputs and float accumulation on 16×16×16 operations, with a uniform loop over K chunks and checked tail padding. Compilation, two host negative controls, and 102 CPU checks on clean source passed. However, 10 of the first 12 device cases failed the predefined exact contract: 177 unequal values, with maximum absolute residual 4.76837158203125e-7. Prepared input files matched the contract, all outputs were finite, and guards were intact. The original reverse run and performance traces remained behind the failed gate and were not launched.

An independent diagnosis reused the same retained binary and inputs for two K16 repetitions, one K64 repetition, and a K16 resource trace. Their complete output bits matched the corresponding original cases, and the unchanged strict checker continued to fail. Independent analysis of all 16 actual cases covered 32768 input halfwords, 4096 C values, and 2048 guards, with 277 unequal values. A CPU replay that rounds every step to FP32 still exactly matched the integer-dot/256 reference; ordinary summation reordering alone cannot explain the difference.

[Numerical diagnosis](../wiki/wmma-exactness.md) · [Complete records retaining the failed status](../data/results/20261008-wmma-exact-diagnostic.json). The official WMMA chapter does not specify a strict IEEE, stepwise FP32 rounding contract. The cause remains unresolved and cannot simply be labeled a hardware defect. The record uses `device-correctness`, `passed=false`, and `performance_accepted=false`. The index gained an explicit correctness-diagnostic route; the original `local-measurement` performance gate still requires `passed=true`. Neither oracle nor tolerance changed.

Five device workers and one profiled application exited and passed release checks. The diagnostic trace contained 110 kernels, reported 28 registers and zero shared/private memory, and had 110 false recompilation flags. These observations do not establish an arithmetic mechanism or performance result. Nothing was promoted to open-cake-ir. Public CI for the earlier `4aa7d4a` release passed.

## 2026-10-08: Device-input control for WMMA and scalar source

Successor source `403a74a` preserved the WMMA function body and strict oracle. It added a scalar-source FP32 implementation reading the same A/B device allocations, with three complete input readbacks: before computation, between implementations, and after computation. Each implementation had its own C buffer and guards; outputs were immediately saved by WMMA/scalar role, independent of execution order. Equal source bodies were not treated as proof of equal binaries.

There were 25 logical cases: 12 forward/WMMA-first, 12 reverse/scalar-first, and one paired K16 trace. All 153600 snapshot halfwords matched prepared/fixed inputs. All 25 scalar output matrices were exact; WMMA still had 384 unequal values: 177 in each sweep and 30 in the trace. Another 12 default mode0 cases served as regression coverage. All 37 WMMA output matrices matched their corresponding old `ef22b51` outputs word for word. Scalar success did not replace WMMA failure.

[Control section](../wiki/wmma-exactness.md#successor-device-input-snapshots-and-a-scalar-fp32-source-control) · [All input readbacks and both output paths](../data/results/20261008-wmma-scalar-control.json). The paired trace was split by actual function names into two 110-event groups, each with its own ten warmups removed. It reported WMMA block64/28 registers and scalar block256/36 registers; both groups reported zero shared/private memory and 110 false recompilation flags. Timing is retained for traceability, without performance acceptance or a speed ratio.

This narrows the investigation to the instrumented WMMA path. The three capture boundaries do not establish transient inputs inside the kernel, so they still cannot uniquely assign the cause to hardware. Clean source passed 121 CPU checks, both local compilation modes, and three host negative controls with devices hidden. Earlier software CI success at `edf9857` did not change WMMA's numerical failure. Four workers and one profiled application exited and passed release checks, with no promotion to open-cake-ir.

## 2026-10-08: Canonical English documentation and navigation

The README, wiki pages, documentation, probe guides, and catalog now use canonical English. Home links and reproduction links make it easier to move between explanations, protocols, and retained evidence. Original results and the failed WMMA status are preserved. These documentation and navigation changes required no GPU execution.

## 2026-10-08: Logical WMMA prefixes and a two-product witness

Frozen source `0cddd51` introduced a separate paired prefix suite: M=N=1 and M=N=16, each with K0–16, while preserving both kernel bodies, the exact integer oracle, full physical participation and input snapshots. Clean-source verification passed 127 CPU checks. All three modes compiled, and nine host negative controls refused invalid families, K, order, padding, headers or case count before device APIs.

Two complete 34-case sweeps used opposite case/implementation orders; singleton K16 and dense K16 traces were preselected. All 70 scalar matrices were exact. WMMA retained 931 unequal elements across 50 failing matrices; 20 matrices were exact. Inputs matched at all three capture boundaries, all outputs were finite and guards intact. Complete outputs matched across orders and corresponding traces. Every K gave identical C[0,0] words across singleton/dense families.

Dense K2 first failed at C[13,2]: the two products 3/64 and -13/256 sum exactly to -1/256, but WMMA returned `0xbb800001` instead of `0xbb800000`. Singleton first failed at K4/C[0,0], which is also the earliest C00 failure for dense. C00 becomes exact again at K6,7,11,12; these are bounded earliest observations, not global minima or a persistent threshold.

[Full prefix curve and witness](../wiki/wmma-exactness.md#successor-logical-k-prefixes-and-a-two-term-witness) · [All actual inputs, snapshots, outputs and batches](../data/results/20261008-wmma-prefix.json). The auxiliary old default/control regressions contribute another 24 logical cases and 36 outputs, all matching their earlier buffers. Overall: 94 logical cases, 176 outputs and 1,760 batches; all 82 scalar outputs exact, WMMA contract still failed, performance still unaccepted. Six workers and two profiled applications exited and passed release checks. No promotion to open-cake-ir.

## 2026-10-08: Isolating and relocating the two-product WMMA witness

Source `6fd66b9` adds a separate three-pattern input contract at M=N=16,K=2. The original dense formulas are followed by an isolated A row13/B column2, then the same ordered pairs relocated to row0/column0. Both kernel bodies and launch helpers remain unchanged. The fixed source passed 136 CPU checks, four native build modes, 15 input/format refusals and three compile-mode refusals before GPU work.

Two opposite-order sweeps and three preselected traces yield nine paired primary observations. Every WMMA matrix has one target mismatch, `0xbb800001` instead of exact `0xbb800000`; scalar matrices are all exact. All other isolated outputs are zero, snapshots agree and guards remain intact. Isolation changes 29 previously nonzero halfwords per operand, and relocation changes four; the ordered pairs remain bitwise identical. Complete outputs match across orders and corresponding traces. The residual therefore survives both interventions, without identifying a unique cause or general position independence.

[Isolation and relocation evidence](../wiki/wmma-exactness.md#successor-isolation-and-relocation-of-the-two-products) · [All retained input/output words and batches](../data/results/20261008-wmma-witness.json). Primary coverage is 9 logical cases and 18 matrices; auxiliary default/control/prefix regressions add 58 logical cases and 104 matrices, matching prior outputs. Overall: 67 logical cases, 122 matrices and 1,220 batches; scalar 55/55 exact, WMMA 14/67 exact with 813 unequal elements. Exact acceptance remains failed and performance unaccepted. All eight workers and three profiled applications exited and passed release checks; no promotion to open-cake-ir.

## Next questions

- Test the isolated C00 products separately, then together in both K orders, under a successor input contract. Preserve physical participation, scalar controls and exact checks. These new combinations have not been measured.
- Inspect generated code and the WMMA path's load/arithmetic implementation. Do not infer ISA behavior from the scalar source label or API fragment type.
- Preserve the original failure, exact oracle, and non-performance diagnostic status. The container Triton route has only package-identity observations so far and cannot inherit qualification from the current native route.
