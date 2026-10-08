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

## 2026-10-08: Standalone products and K-slot permutations

Source `7b8ee81` explicitly extends the probe selector with mode 2 for a new six-pattern product contract; older mode 0/1 suites remain closed. Logical M=N=16,K=2 and C00 are fixed. The two product values are tested separately at K0 and K1, then together in both slot assignments. Kernel and launch bodies remain unchanged. The frozen source passed 145 CPU checks, five native builds, 18 input/format refusals and three incompatible-mode refusals.

Two opposite-order six-case sweeps and six preselected traces retain 18 paired observations. All 12 standalone-product WMMA matrices are exact; the six paired WMMA matrices each have one C00 mismatch, always `0xbb800001` instead of `0xbb800000`. All 18 scalar matrices are exact. Other outputs are zero, all snapshots agree and guards remain intact. Complete outputs agree across the tested slots, pair permutations, execution orders and corresponding traces. The exact CPU sum of the observed component values is -1/256, while the joint WMMA value differs by -2^-31. This does not identify an internal accumulation order or hardware cause.

[Component and K-slot evidence](../wiki/wmma-exactness.md#successor-single-products-and-k-slot-permutations) · [All retained words and batches](../data/results/20261008-wmma-products.json). Auxiliary default/control/prefix/witness regressions contribute 61 logical cases and 110 output matrices, all matching prior buffers. Overall: 79 logical cases, 146 matrices, 1,460 batches; scalar 67/67 exact and WMMA 26/79 exact with 813 unequal elements. Correctness remains failed and performance unaccepted. All 12 workers and six profiled applications exited and passed release checks; no promotion to open-cake-ir.

## 2026-10-08: Fixed-magnitude sign configurations

Source `d1e1be9` adds a separate mode 3 sign contract while preserving older modes. It fixes M=N=16,K=2,C00 and joint B numerators [-1,-13], varying A signs to obtain all four combinations of product numerators 12 and 13 on the /256 output scale. Four standalone controls zero both inactive operands. Device and launch bodies remain unchanged. The frozen source passed 154 CPU checks, six native builds, 21 input/format negatives and three incompatible-mode negatives.

Two eight-case sweeps in opposite orders and eight preselected traces retain 24 paired observations. Standalone and same-sign conditions are exact; the six mixed-sign WMMA matrices have one C00 mismatch each. `pair-pn` returns `0xbb800001` versus `0xbb800000`; `pair-np` returns `0x3b7ffffe` versus `0x3b800000`. Both signed residuals are -2^-31, although their adjacent-FP32 distances are one and two steps. All 24 scalar matrices are exact; other primary outputs are zero, snapshots agree and guards remain intact. These input contrasts do not establish an internal rounding or accumulation mechanism.

[Sign-configuration evidence](../wiki/wmma-exactness.md#successor-fixed-magnitude-sign-configurations) · [All retained words and batches](../data/results/20261008-wmma-signs.json). Five auxiliary suites contribute 67 logical cases and 122 matrices, all matching prior buffers. The three explicitly input-matched primary conditions add 18 matching historical variant comparisons. Overall: 91 logical cases, 170 matrices and 1,700 batches; scalar 79/79 exact, WMMA 36/91 exact with 815 unequal elements. Correctness remains failed and performance unaccepted. All 15 workers and eight profiled applications exited and passed release checks; no promotion to open-cake-ir.

## 2026-10-08: Adjacent product magnitudes at a fixed result

Source `9b6c28c` adds mode 4 for a closed 42-condition magnitude contract: every integer q from 1 through 14, each with positive, negative and paired roles. All hold M=N=16,K=2,C00 fixed. Paired A=[-q,1]/16 and B=[-1,-(q+1)]/16 give exact sum -1/256; component cases zero both operands of the inactive term. Kernel and launch bodies remain unchanged. The frozen source passed 163 CPU checks, seven native builds, 26 input/format negatives and three incompatible-mode negatives.

Two complete sweeps use opposite case/implementation orders. Both standalone components are exact at every q. Paired WMMA is exact at q1–6 and has only C00 unequal at q7–14, always `0xbb800001` versus exact `0xbb800000`, with signed residual -2^-31. All scalar matrices are exact, other primary outputs are zero, snapshots agree and guards remain intact. The fixed final reference does not identify this response: product magnitudes, cancellation severity and operand encodings change together. The first observed paired mismatch at q7 applies only to the declared grid and factorization.

Only three preselected pair conditions have primary traces: q1, q12 and q14. Their buffers match both sweeps; no standalone or q6/q7 boundary trace is claimed. [Full grid and interpretation](../wiki/wmma-exactness.md#successor-adjacent-product-magnitudes-at-a-fixed-result) · [All retained words and batches](../data/results/20261008-wmma-magnitudes.json). Primary coverage is 87 logical cases and 174 matrices: WMMA 69/87 exact with 18 mismatches, scalar 87/87 exact. Six auxiliary suites contribute 75 logical cases and 138 matrices. All 138 auxiliary and 14 input-matched q12 historical comparisons agree.

Overall: 162 logical cases, 312 matrices and 3,120 batches; scalar 150/150 exact, WMMA 93/162 exact with 829 unequal elements. Numerical acceptance remains failed and performance unaccepted. All 11 workers and three profiled applications exited and passed release checks; no promotion to open-cake-ir.

## 2026-10-08: Exact power-of-two A scaling

Source `2725811` adds a closed dyadic scale contract at q6/q7/q12 and e=-2,-1,0,1,2, each with positive, negative and paired roles. A alone is multiplied by 2^e; B, K slots and the physical tile stay fixed. Pair references are -2^e/256. The 45 parameter conditions contain 41 distinct complete input pairs because four positive controls coincide across q/scale settings. All old modes and device/launch bodies remain unchanged.

Both full opposite-order sweeps find every standalone control exact. q6 pairs are exact at all five scales; q7 and q12 each have one C00 mismatch with signed residual -2^(e-31). Dividing by 2^e gives -2^-31 throughout these failing cases, and each value is one adjacent FP32 step below its reference. Absolute error varies with scale. All scalar matrices are exact. These finite observations do not identify a rounding mode, internal precision, general scaling law or unique cause. The exact CPU sum of observed component targets matches the reference; it is not a separate GPU addition measurement.

Only q7 pairs at e=-2,0,+2 were profiled. Their buffers match both sweeps, as do all input-alias comparisons. [Full scale table and interpretation](../wiki/wmma-exactness.md#successor-exact-power-of-two-scaling-of-a) · [All actual words and batches](../data/results/20261008-wmma-scales.json). Primary coverage is 93 logical cases and 186 matrices: WMMA70/93 exact with 23 mismatches, scalar93/93 exact. Seven auxiliary suites add 117 logical cases and 222 matrices. All 222 auxiliary and 58 input-matched primary historical comparisons agree, including matches at nonzero exponents.

Overall: 210 logical cases, 408 matrices and 4,080 batches; WMMA128/210 exact with 842 unequal elements, scalar198/198 exact. All snapshots agree, outputs are finite and guards intact. Correctness acceptance remains failed and performance unaccepted. All 12 device workers and three profiled applications exited and passed release checks; no promotion to open-cake-ir.

Frozen source passed 172 CPU tests and a separate host C++ encoder check of 155 inputs/four refusals. All eight native builds, 31 input/format negatives and three mode negatives passed before GPU work. The original CPU-build SSH connection closed with exit255; inspection found five completed builds, no compiler processes and three untouched modes. A separate continuation built only those three, preserving all earlier records and the unknown original driver's exit/cause. No GPU stage required a retry.

## 2026-10-09: Reciprocal exponents at fixed products

Source `b1eefc4` adds a closed reciprocal contract: q6/q7/q12, A exponent e=-2,-1,0,1,2, B exponent -e, and positive/negative/paired roles. All 45 parameter conditions have distinct complete inputs. Every exact product and the paired reference -1/256 stay fixed while operand exponents change. Both exponent fields are bound explicitly; old modes, encoders and device/launch bodies remain unchanged. The run ID `20261008-wmma-reciprocal` uses the UTC date.

Both full opposite-order sweeps find every standalone component and scalar matrix exact. q6 pairs are exact; q7/q12 pairs each have only C00 unequal, always `0xbb800001` versus `0xbb800000`, with residual -2^-31 and adjacent-FP32 distance 1. For each fixed q/role/implementation/order, complete buffers are identical across all five exponent pairs. Corresponding outputs match across orders and all three traces. This bounded invariance does not identify the arithmetic cause or establish behavior under arbitrary factorization or sign placement. The fixed-reference residual is not divided by an operand's scale.

Only q7 paired e=-2,0,+2 were profiled. [Full reciprocal results](../wiki/wmma-exactness.md#successor-reciprocal-exponents-with-fixed-products) · [All actual words and batches](../data/results/20261008-wmma-reciprocal.json). Primary coverage is 93 logical observations/186 matrices: WMMA 70/93 exact with 23 mismatches, scalar 93/93 exact. Eight auxiliary suites add 162 logical cases / 312 matrices. Nine primary conditions form 11 links to older scale patterns; 38 measured primary variant outputs generate 46 historical comparison rows because of prior aliases. With 312 auxiliary rows, all 358 comparisons agree and cover 350 current variant outputs. Comparison counts are not extra measurements.

Overall: 255 logical cases, 498 matrices and 4,980 batches; WMMA 163/255 exact with 852 unequal elements, scalar 243/243 exact. All snapshots agree, outputs are finite and guards intact. Correctness acceptance remains failed and performance unaccepted. All 13 device workers and three profiled applications exited and passed release checks; no promotion to open-cake-ir.

Frozen source passed 181 CPU tests. A separate host-only compilation of its metadata emitter matched all 15 protocol fields and 45 pattern records, including required plural lists. Nine native modes compiled in individual retained stages; 38 input/format negatives and three mode negatives passed before GPU admission. No compilation or device stage was retried.

## 2026-10-09: Product-preserving factor-sign transfers

Source `747c7b5` adds 24 distinct sign-transfer inputs at q6/q7/q12: two positive-component placements, two negative-component placements and four paired placements per q. A transfer negates both A and B at the selected term, keeping the signed product and reference fixed. Inactive flags must stay zero and both inactive operands remain positive zero. Both kernel/launch bodies, encoders and older contracts remain unchanged. The run ID `20261008-wmma-sign-transfer` uses the UTC date.

Both full opposite-order sweeps find all components and scalar matrices exact. q6 pairs are exact; q7/q12 pairs have only C00 unequal in all four placements, always `0xbb800001` versus `0xbb800000`, with signed residual -2^-31 and adjacent-FP32 distance 1. Full output buffers remain equal within each fixed q/role across placements, across orders and against the corresponding traces; changed operand sign words are retained. This does not establish arbitrary factorization invariance or locate the arithmetic cause. Each pair is compared with its matching positive and negative component observations, without counting reused component values as new measurements.

Only the four q7 paired placements were profiled. [Full sign-transfer evidence](../wiki/wmma-exactness.md#successor-transferring-factor-signs-at-fixed-products) · [All actual words and batches](../data/results/20261008-wmma-sign-transfer.json). Primary coverage is 52 logical observations and 104 matrices: WMMA 32/52 exact with 20 mismatches, scalar 52/52 exact. Nine auxiliary suites add 207 logical cases and 402 matrices. All 402 auxiliary and 38 input-matched primary historical comparisons agree, covering 440 current variant outputs.

Overall: 259 logical cases, 506 matrices and 5,060 batches; WMMA 160/259 exact with 859 unequal elements, scalar 247/247 exact. All snapshots agree, values are finite and guards intact. Exact acceptance remains failed and performance unaccepted. All 15 device workers and four profiled applications exited and passed release checks; no promotion to open-cake-ir.

Frozen probe source passed 190 CPU tests. Host-only compilation of the exact metadata emitter matched 12 protocol fields and all 24 records. Ten native modes compiled in separate retained CPU stages; 42 input/format negatives and three incompatible-mode negatives passed before device admission. No compile or device stage was retried.

## 2026-10-09: Concise retrieval from the canonical English page

The WMMA page now begins with a Current findings table linking to each detailed control. `python3 scripts/wiki.py show c550-wmma-exactness --summary` reads that marked block directly from the page; the catalog remains the sole retrieval index. Default `show` still returns the complete page. Missing, malformed or empty summary blocks fail explicitly rather than falling back to the long text. Existing historical sections, failed outcomes and anchors remain present. The new retrieval behavior has seven temporary-fixture CLI tests and is separate from frozen device source `747c7b5`.

## 2026-10-09: Standalone single-launch q7 reproduction

Source `9b7bef6` adds a separate 241-line C++ collector and a standard-library Python checker for the fixed q7 input. The two device bodies match frozen `747c7b5`; the host package no longer uses earlier suite tables or an input plan. A/B each retain 1,024 halfwords, one padded K16 step, full 64/256-thread WMMA/scalar blocks, separate guarded outputs and three complete input readbacks. The successor protocol launches each implementation once, with zero warmups and no collector event timing. The run ID `20261008-wmma-q7-repro` uses the UTC date.

WMMA-first, scalar-first and a separate WMMA-first trace all return WMMA C00 `0xbb800001` versus exact `0xbb800000`, with signed residual -2^-31 and one adjacent FP32 step. All three scalar matrices are exact; the other 255 outputs are zero, values finite and guards intact. All 18,432 snapshot halfwords agree. After full prepared-input matching, all six guarded outputs equal the canonical q7 f00 observation from the previous sign-transfer WMMA-first sweep. The new checker retains three `numeric_failed` exits of 1; it never substitutes the observed failure word for the reference.

[Small reproduction guide](../experiments/wmma_q7/README.md) · [Measured outcome and protocol boundary](../wiki/wmma-exactness.md#standalone-q7-reproducer-one-launch-per-variant) · [Complete retained words](../data/results/20261008-wmma-q7-repro.json). Coverage is three paired collections of one fixed input, six matrices, 6,144 prepared halfwords, 18,432 snapshots, 1,536 payload values and 768 guards. Six kernel launches are declared overall; only the two events in the traced process are directly profiled. The profile applies no warmup removal and reports WMMA64/28 registers and scalar256/36, zero shared/private memory and false recompilation flags. Profiler units remain unverified; there are no timing batches or accepted performance claims.

Frozen source passed 209 CPU tests, including 12 new tests. A host-only extraction matched the actual metadata output statements for both orders, using synthetic observed fields. The single native build and four pre-device CLI/output-directory refusals passed. All three workers and the profiled application exited and passed release checks. Old probe source files were unchanged, and no earlier GPU suite was rerun for this package. No result was promoted to open-cake-ir.

This separately built one-launch collector reproduces the mismatch on the tested stack. It does not establish a cold device, identical binary or isolated arithmetic cause. Its retained executable can support subsequent CPU-only inspection or a separately qualified compiler/runtime comparison; newly recompiled IR must remain distinct from the executed artifact.

## 2026-10-09: Inspecting the retained q7 binary

The executable from source `9b7bef6` and run `20261008-wmma-q7-repro` was inspected without rebuilding or executing it. Its 34,629-byte `.mc_fatbin` contains an empty host entry, 11,216 bytes of wrapped LLVM bitcode and an 18,232-byte native device ELF, both labeled `xcore1000`. SDK extraction preserved the original executable and matched each declared byte range. Independent bounded parsing and 14 extracted-copy comparisons agreed. The new inspection ID uses the UTC date, `20261008-wmma-q7-binary-inspection`.

SDK `llvm-dis` decoded the actual packaged bitcode. WMMA has five static MMA intrinsic sites across remainder and unrolled loops; scalar has eight static `fmul contract`/`fadd contract` pairs. These are IR observations, not runtime operation counts, native instruction identities or precision guarantees. Native metadata reports 28/36 mtregs and a 512-thread block attribute; it does not establish device ceilings or occupancy. System objdump could not decode the device ELF, and the bounded SDK/PATH search found none of the tested decoder names.

[Inspection guide and verified commands](compiled-artifacts.md) · [Selected static observations](../data/inspections/20261008-q7-binary.json). Runtime payload selection and arithmetic cause remain unresolved. No GPU lease, new device execution, numerical acceptance, performance measurement or open-cake-ir promotion occurred. Full binaries and IR remain private retained evidence. Public English navigation now links the inspection, while the toolchain guide points readers to the canonical WMMA findings instead of repeating the full chronology.

A fresh check of the official release history corrects the documentation's earlier `-offload-arch=native` attribution: it appears under SDK 3.5.3.18, whose stated product group includes C550. The reproduction template still uses an explicit, locally evidenced architecture; this source correction adds no compiler execution or device qualification.

## 2026-10-09: q7 through an explicitly supplied native ELF

Source `b463205` adds a host-only module collector with no embedded device fatbin. It supplies only the 18,232-byte native ELF extracted from the retained `9b7bef6` executable, loads it through `mcModuleLoadData`, looks up the original symbols and passes four argument addresses to `mcModuleLaunchKernel` with null `extra`. The installed sample and pinned mcTriton use this convention despite the 3.5.3 API warning; its operational acceptance was tested rather than inferred. The run ID `20261008-wmma-module-q7` uses the UTC date.

WMMA-first, scalar-first and a separate resource trace all load, launch and unload successfully. Each WMMA output has only C00 unequal: `0xbb800001` versus exact `0xbb800000`, residual -2^-31. All three scalar matrices are exact. Independent decoding checks all 26,880 prepared/snapshot/payload/guard words; every captured input agrees, payloads are finite and guards intact. After complete input matching, all six guarded outputs equal their corresponding earlier q7 collection. Each retained checker also verifies the supplied image against the independent extraction at that handoff boundary. Exact acceptance remains failed and no performance is accepted.

[Module reproduction guide](../experiments/wmma_module/README.md) · [Bounded conclusion](../wiki/wmma-exactness.md#successor-explicit-native-elf-module-loading) · [Complete numerical record](../data/results/20261008-wmma-module-q7.json). The trace contains exactly two events in the declared order, with blocks 64/256, registers 28/36, zero shared/private fields and false recompilation descriptors. Raw durations retain unverified units. The attempted `mcTracer --version` query returned zero with an `execvpe` error and established no build version; this failed metadata attempt remains recorded. `MACA_MODULE_LOADING` was not captured.

The frozen source passed 221 CPU tests, including 12 new module-checker tests. A host-only GNU C++ build and seven pre-device argument/image refusals passed. Factoring the shared q7 file analysis preserves the three old checker reports exactly. All three workers and one profiled application exited with verified release before CPU checking. No new device-source compilation or open-cake-ir promotion occurred.

The native image can therefore reproduce the residual when explicitly supplied. This does not determine the older fatbin's selected entry, rule out loader/cache transformations or explain native arithmetic. The public English page retains that distinction and the original failures.

## 2026-10-09: A generated bitcode-only carrier is refused at module loading

Source `3a08040` adds explicit native/retained-bitcode image kinds to the shared module collector and a v2 protocol. It retains one q7 oracle and one collection/launch path, requires independent carrier/payload references, and records `MACA_MODULE_LOADING`. Earlier v1 evidence still replays at `b463205`. The installed bundler first generated a 15,324-byte carrier with two descriptors: an empty host entry and the exact original 11,216-byte wrapped bitcode. Its 12-byte `__FILE_END__` trailer has no NUL. CPU preparation verifies the actual layout and whole payload; no device source or LLVM IR is recompiled. The preparation/run IDs use the UTC date.

The native-v2 control completes with the previous single WMMA C00 residual (-2^-31) and exact scalar output. Both complete guarded matrices match the corresponding earlier native-module collection after complete prepared-input matching. The first bitcode-carrier attempt then fails at `mcModuleLoadData`, returning `mcErrorNoKernelImageForDevice`; the runtime diagnostic names a missing `xcore1002` binary. The collector stops before function lookup, its device allocations/uploads, snapshots or kernel launch. The requested scalar-first and trace stages remain untouched. No format fallback, retagging or retry occurs.

[Full route observation](../data/results/20261008-wmma-bitcode-route.json) · [Bounded interpretation](../wiki/wmma-exactness.md#successor-bitcode-only-carrier-rejected-before-execution) · [v2 reproduction guide](../experiments/wmma_module/README.md). The failed attempt contains only its supplied image, host-prepared A/B files and one device record. A post-release audit matches the entire supplied carrier with the prepared reference and its payload with the original bitcode. This proves the input boundary, not successful module admission. Bitcode numerical evaluation is **not reached**; it is not a failed or passed numerical comparison. The prior native numerical result remains the catalog's primary result.

Independent analysis decodes 8,960 native-control words and 2,048 failed-attempt host halfwords. Native input snapshots, finite payloads and guards pass. Both workers, lock-PID observations and observed process groups pass release checks; no profile ran. The generic allocator release-boundary template is not used as proof that the failed collector synchronized or retained GPU outputs. The successful collector reports `MACA_MODULE_LOADING` unset; the failed attempt retains only its requested environment, also unset.

Frozen source passed 229 CPU tests. Both actual input forms passed the public CPU inspection gate, and 12 pre-device host-binary refusals passed. The empty cache listings do not establish the absence of internal runtime work. The loader diagnostic does not qualify its suggested compiler flag, reject all bitcode forms or explain which entry/format/JIT rule rejected this carrier. Nothing was promoted to open-cake-ir, and no performance is accepted.

## 2026-10-09: MCRTC producer serialization observed without module loading

Source `83f6384` adds a standalone host API probe for an independently authored empty kernel. It uses the installed `mcr/mcrtc.h` declarations and `libmcruntime.so` exports, zero headers/options and null pointers as permitted by the installed interface. All four visibility masks are required present and empty before its first MCRTC call. It calls no device enumeration, module-loading or launch API. The library's internal device/context behavior is not profiled. The run ID `20261008-mcrtc-format` uses the UTC date.

The valid source returns producer exit 0 and a **7,360-byte LLVM bitcode wrapper**. Its fields are version 0, body offset 20, body size 7,332 and raw CPU type 255, followed by eight zero bytes. The installed `llvm-dis` 19.1.3 decodes the retained bytes directly. The IR contains `mcrtc_format_probe` as `metaxgpu_kernel`, triple `mxc-metax-macahca`, and `xcore1000` CPU/features for this case. Decoded `source_filename` is `ld-temp.o`; the recorded source/API/output sequence owns source attribution. No q7 source or old carrier is rebuilt.

The version API reports integers 1/0, separately from the SDK directory and host compiler identity. The positive log is one NUL byte. The independent `#error` case returns API compilation code 6 and producer exit 1, with a 207-byte NUL-terminated log containing its marker and no output buffer. The negative remains `compile-failed`; checker exit 0 only confirms that declared control. Eight and six status records are retained respectively; error-string helper calls are additional SDK invocations. Both destruction calls succeed and both observed producer process groups are empty after exit.

[Producer guide](../experiments/mcrtc_format/README.md) · [Format comparison](compiled-artifacts.md#mcrtc-returns-a-wrapped-bitcode-buffer-in-a-separate-cpu-probe) · [Selected API and serialization observation](../data/inspections/20261008-mcrtc-producer.json). Frozen source passed 241 CPU tests; the actual host build and 12 pre-API refusals passed. No GPU lease, module-load or kernel-launch command was submitted. No numerical or performance acceptance and no open-cake-ir promotion occurred.

This producer output has a different outer form from the rejected two-entry Clang carrier. It shares a wrapper convention with the older extracted q7 bitcode, but the source programs and byte extents differ. The observation supplies no q7 byte identity, old fatbin selection or loader/JIT acceptance. The earlier load refusal and numerical failures remain preserved.

## Next questions

- Establish the earlier fatbin's payload selection or obtain evidence of the final runtime instructions for this C550/SDK pair. The explicit native-module result establishes its own route and does not settle either question.
- Use a separately frozen protocol to test explicit wrapped-bitcode inputs, keeping producer-origin and q7 artifacts distinct. The observed MCRTC format motivates a candidate input form; it does not qualify loading or execution. Preserve the generated-carrier refusal and distinguish producer format, target-entry selection and numerical behavior.
- Qualify any independent SDK/compiler environment and its allocation/release lifecycle before a numerical comparison. Existing container package or requested-mount metadata alone is insufficient; preserve installed SDKs and production containers.
- Preserve the original failures, exact oracle and non-performance diagnostic status. Neither a float fragment type nor the scalar source label identifies the native arithmetic mechanism.
