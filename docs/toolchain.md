# C550 toolchain: native targets and compatibility interfaces

[Home](../README.md)

This page distinguishes online source code, official documentation, and the local installation. See the [source index](sources.md) for primary references and their scope. Experiment records own C550 measurement results.

## Environment-record scope

A read-only inspection of C550-1 on 2026-10-07 reported the host installation at `/opt/maca-3.5.3` and MXCC version `1.0.0 (6477545d4d)`. Retain both identifiers: an SDK path and a compiler report are different facts. Reproduction records must also retain driver, runtime, container, and Python package versions, plus the absolute path of the compiler actually invoked.

The installed `mc_runtime_api.h` defines a compatibility macro from `waveSize` to `warpSize`, while type declarations contain both names. The official runtime guide's query example uses `waveSize`. Two spellings do not establish two independent hardware facts; retain the device-query source, include paths, and output together. This section does not supply property values absent from experiment records.

## Three development interfaces

| Interface | Purpose and evidence to retain |
| --- | --- |
| MXCC / MXMACA C++ | Direct MACA runtime and kernel extensions; retain the native target, complete command, generated device code, and runtime errors |
| cu-bridge / cucc | Adapts CUDA-style source to MACA; the official tutorial uses `/opt/maca/tools/cu-bridge/bin/cucc`. Retain the actual path and expanded compilation command |
| mcTriton | A Python/Triton frontend that generates device code through the MetaX backend; retain wheel version, backend file locations, target, options, and artifacts |

The native interface is documented in the [runtime guide 3.5.3.x][runtime], and the compatibility interface in the [official vector-add tutorial][vector-guide]. The [older runtime-guide URL][runtime-legacy] used initially showed active version 3.0.0.x on 2026-10-07; this page now cites the rechecked 3.5.3.x version. The mcTriton implementation observations below are pinned to [source commit `7dd407c`][triton-readme], which has not been established as the source of the installed C550 package.

## Native compilation template

The official documentation uses `mxcc -x maca` with an explicit MACA path. This template requires a native architecture supported by local evidence; it is not a passed test record:

```sh
export MACA_PATH=/opt/maca-3.5.3
export LD_LIBRARY_PATH="$MACA_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
: "${C550_ARCH:?Set the native architecture from local device and compilation evidence}"
"$MACA_PATH/mxgpu_llvm/bin/mxcc" \
  -x maca -O3 --offload-arch="$C550_ARCH" \
  --maca-path="$MACA_PATH" probe.cpp -o probe
```

Check the spelling and supported values of `--offload-arch` against local `mxcc --help` and successful build logs. The online [release notes][release-changes] list `-offload-arch=native` among the 3.7.0 changes; it is therefore not this page's default for the 3.5.3 host. Do not derive the native target from `torch.cuda.get_device_capability()` or Triton's `arch=80`.

## The 512-thread function attribute and runtime recompilation

[Runtime guide 3.5.3.x §4.3][runtime-cache] gives a vector-add example with no `__launch_bounds__` annotation and 1024 threads per block. It states that this example triggers recompilation and binary caching because the block exceeds 512 threads. This motivates a local comparison between the default compiled function and its larger-block runtime path. It does not make the trace field `max_block_size=512` a device limit or guarantee that 1024 threads are faster.

[The same version's §4.4][runtime-env] defines `MACA_CACHE_PATH` as the binary-cache directory, defaulting to `.metax/shadercache/` under the user's home directory. `MACA_CACHE_DISABLE=1` disables that cache; 0 or an unset variable enables it. A new run can use an isolated cache directory and retain directory listings, function attributes, full outputs, and host timing before and after its first execution, then launch later processes against that same directory. File changes and first-launch delay are clues for a recompilation hypothesis; runtime logs or artifacts are needed to establish compilation directly. This binary cache is a compilation cache, not an L2/HBM cache-control mechanism.

The public-document search in this round did not find a complete `__launch_bounds__` definition applicable to the current MXCC, particularly treatment of its second parameter and behavior above an explicit first parameter. Retain installed headers, compilation diagnostics, and controlled execution evidence before drawing conclusions. Do not substitute CUDA or HIP definitions.

<a id="动态-shared-容量的后继控制入口"></a>

## Dynamic shared-memory capacity as a follow-up control

The official [Runtime API reference 3.5.3.x entry for `mcLaunchKernel`][dynamic-launch] defines `sharedMemBytes` as the dynamic shared-memory bytes requested for that launch and supports external shared declarations. The [programming guide][extern-shared] provides the `extern __shared__` syntax. These interfaces support a same-kernel, same-pitch comparison that changes only the requested dynamic reservation, subject to local compilation, resource observations, and correctness checks.

The [attribute reference][dynamic-attributes] distinguishes static shared usage from the maximum permitted dynamic allocation: the static value excludes the launch's dynamic request, and the maximum is not the actual request. The [limit-setting interface][dynamic-limit] requires the dynamic maximum plus static shared memory to remain within the device-declared per-block limit. Do not import a CUDA default of 48 KiB or treat requested capacity or occupancy-API predictions as measured residency.

<a id="64-lane-collective-的接口与实测范围"></a>

## 64-lane collective contracts and measured scope

On 2026-10-08, the official C++ guide's active version selector was verified as 3.5.3.x. Its [shuffle section][cpp-shuffle] declares `__shfl_sync(unsigned long mask, T value, int srcLane, int width=warpSize)`: width is a valid power-of-two subgroup width, and direct source indices beyond that width select `srcLane % width` within the subgroup. Masked threads must be active and make matching calls; the source thread must also participate. The [integer reduction section][cpp-reduce] declares `__reduce_add_sync(unsigned long mask, int value)` and its overload for an unsigned-int value.

This repository has checked direct shuffle and integer sums on complete 64/128-thread blocks, saving every participating thread's outputs. Logical tails supply zero and no thread exits early. The [wave64 results](../wiki/wave-collectives.md) specify the tested scope and retain every actual output. The [synchronization section][cpp-sync] separately states the memory-order guarantee of `__syncwarp`; register shuffle is not a substitute for shared-memory synchronization.

The official page remains a family-level interface contract. The local investigation separately checked the 64-bit mask type, installed headers, compilation, and execution. Inconsistencies in down-shuffle boundary wording, XOR descriptions, and vote-mask types were not used to infer untested behavior. An [independent follow-up](../wiki/wave-collectives.md#successor-full-typed-masks-select-different-reduction-interfaces) tested the installed mask-type overloads: the current SDK's full unsigned 32-bit mask operates on 32-element groups, while its full unsigned-long 64-bit mask operates on 64-element groups. The physical wave remains 64. Other mask values, participation patterns, and software versions are outside that result.

## Native WMMA interface and observed numerical limits

The local 3.5.3 compiler resource header `__clang_maca_mma_functions.h` provides `mxmaca::wmma`. The official example's `mma.h` is on the cu-bridge path, so the include chain must be checked. The current FP16-to-float-accumulator overload uses four-argument `mma_sync`; do not invent a `satf` parameter from the generic description. The native `-x maca` probe compiled and ran, but failed its strict exact-dyadic contract; see the [complete numerical diagnosis](../wiki/wmma-exactness.md). A float fragment type does not establish internal rounding guarantees. The [device-input readback and scalar follow-up](../wiki/wmma-exactness.md#successor-device-input-snapshots-and-a-scalar-fp32-source-control) observed matching inputs at the captured boundaries, exact scalar results, and recurring WMMA residuals. This narrows the investigation to the tested WMMA path without uniquely attributing the cause to hardware.

The later [logical-prefix scan](../wiki/wmma-exactness.md#successor-logical-k-prefixes-and-a-two-term-witness) retains the full physical tile and scalar controls while varying logical M/N/K. It exposes a two-product witness at dense K2. A subsequent [isolation/relocation control](../wiki/wmma-exactness.md#successor-isolation-and-relocation-of-the-two-products) retains the same residual after clearing unrelated operands and moving the target. The [component/K-slot controls](../wiki/wmma-exactness.md#successor-single-products-and-k-slot-permutations) find the individual tested products exact, with the same residual in both paired placements. The [fixed-magnitude sign controls](../wiki/wmma-exactness.md#successor-fixed-magnitude-sign-configurations) add exact same-sign cases and two mixed-sign residuals with the same negative offset. The [fixed-result magnitude scan](../wiki/wmma-exactness.md#successor-adjacent-product-magnitudes-at-a-fixed-result) then tests every integer q from 1 through 14 with paired products `(q, -(q+1))/256`. The exact sum stays -1/256: q1–6 are exact and q7–14 have one C00 residual of -2^-31, with exact components and scalar controls throughout. This finite-grid result does not establish a universal threshold or a unique hardware attribution. The [A-scaling follow-up](../wiki/wmma-exactness.md#successor-exact-power-of-two-scaling-of-a) tests q6/q7/q12 at five exact powers of two. q6 stays exact, while q7/q12 retain one adjacent-FP32-step residual, proportional to the tested scale. This finite pattern does not identify internal precision or imply a general scaling law. A [reciprocal-exponent control](../wiki/wmma-exactness.md#successor-reciprocal-exponents-with-fixed-products) holds each signed product and the reference fixed while redistributing exponent between A and B. Complete outputs stay unchanged across the five tested exponent pairs for each q/role; arbitrary factorization or sign-placement invariance remains unestablished.

A separate CPU-only metadata query found Torch 2.10.0 and Triton 3.6.0 with the package suffix `metax3.8.0.4.c600u` in an existing containerd environment. That is not the host 3.5.3 test environment. This wiki has not executed or qualified a matrix kernel in that container. The online mcTriton 3.0 observations below likewise do not replace inspection of that installation's backend.

## mcTriton source observations

The [Python driver][triton-driver] keeps the backend name `maca` and uses 64-lane groups; its launcher computes block threads as `64 * num_warps`. The [C driver][triton-driver-c] separately maps device `major=10/15/16` to compatibility capabilities `80/86/89`. These are interface choices in that source version, not NVIDIA compute capabilities or C550 native ISA identifiers.

The stages in [compiler.py][triton-compiler] are:

```text
Triton → TTIR → TTGIR → LLVM dialect MLIR → LLVM IR → mcfatbin
```

The source admits `num_warps` values 1, 2, 4, 8, and 16, with default 4. It declares default `num_stages=3` and pipelines including `basic` and `cpasync`. Passing Python option checks establishes software admission only. Compilation, correctness, and performance still need separate checks for each shape, type, and pipeline. Start with default options to establish a reproducible baseline.

[triton_metax.cc][triton-codegen] invokes MXCC's `--fatbin` route with `-maca-link -input-is-device` and links MACA bitcode libraries. Source-level diagnostic switches include:

```sh
TRITON_PRINT_COMPILE_OPTIONS=1
TRITON_COMPILER_DUMP_ALL=1
```

The first prints compilation commands; the second adds `--keep` to MXCC. They work only if the installed backend retains these implementations. Keep caches and intermediate artifacts in the current experiment directory so another target's old cache is not mistaken for a new compilation result. The online README's build process requires a separate `metax_llvm` package; this page does not require replacing the installed SDK or rebuilding an existing wheel.

## Timing and profiler evidence

MACA provides event-timing APIs. The official example places start/end events around a kernel, waits for completion, then reads their interval; [vLLM-metax's timing wrapper][kernel-timer] uses the same route. State batching, warmup count, synchronization points, and whether input preparation or output transfer is timed. Retain every sample. An amortized event time for a short kernel is not a pure instruction latency.

Describe cache reset separately. Until the reset method is verified, label the cache state uncontrolled or state the actual warm-reuse policy; do not claim cold-L2 or HBM measurements. Driver optimization notes do not establish the cache state of the current process.

The [mcProfiler manual][profiler] provides a counter-collection entry point; [mxvs][mxvs] covers device, link, memory, and compute tools. This source review has not validated the installed C550 profiler's CLI, counters, permissions, or capture results. Retain tool version, original help, commands, metric definitions, and successful output when testing it. If unavailable, report the coverage gap and preserve the correctness and timing records.

The [mcTracer 3.5.3.x capture chapter][tracer] describes JSON output, and the [Viewer chapter][tracer-viewer] describes opening it in the dedicated UI. Neither inspected chapter supplied a unit contract for exported JSON `ts` / `dur`. Comments on upstream timestamps in installed MCPTI headers do not establish whether the exporter transforms their units. Retain raw values and an unverified-unit label until the export implementation or an explicit contract is available. See the [profiling page](../wiki/profiling.md) for the captured trace's scope and content checks.

[runtime]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/index.html
[runtime-legacy]: https://developer.metax-tech.com/api/client/document/preview/567/C500_RuntimeAPIProgrammingGuide_CN.html
[runtime-cache]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编译和调试.html#binary-cache
[runtime-env]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编译和调试.html#irs9wigbg1oh1
[vector-guide]: https://gitee.com/metax-maca/mxmaca-performance-tuning-guide/blob/main/guide/ch1.初探异构编程.vectoradd.md
[release-changes]: https://developer.metax-tech.com/api/client/document/preview/发布说明/MXMACA_发布说明/曦云C500系列/latest/split_files/新增特性及变更.html
[triton-readme]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/README.md
[triton-driver]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.py
[triton-driver-c]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.c
[triton-compiler]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/compiler.py
[triton-codegen]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/triton_metax.cc
[kernel-timer]: https://github.com/MetaX-MACA/vLLM-metax/blob/f2fcc59c314f1fbc7897f46d465e5f7c35900e8a/csrc/libtorch_stable/quantization/awq/hgemv_selector.hpp
[profiler]: https://developer.metax-tech.com/api/client/document/file/211/preview/?file_type=pdf
[tracer]: https://developer.metax-tech.com/api/client/document/preview/性能测试及分析工具/mcTracer使用手册/曦云C500系列/3.5.3.x/split_files/mctracer.html
[tracer-viewer]: https://developer.metax-tech.com/api/client/document/preview/性能测试及分析工具/mcTracer使用手册/曦云C500系列/3.5.3.x/split_files/mctracer_viewer.html
[mxvs]: https://developer.metax-tech.com/api/client/document/preview/996/index.html

[dynamic-launch]: https://developer.metax-tech.com/api/client/document/preview/990/split_files/mxmaca_运行时api模块.html#mcerror-t-mclaunchkernel-const-void-function-address-dim3-numblocks-dim3-dimblocks-void-args-size-t-sharedmembytes-dparm0-mcstream-t-stream-dparm0
[extern-shared]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编程接口.html#sxjmjcedu7c41
[dynamic-attributes]: https://developer.metax-tech.com/api/client/document/preview/990/split_files/mxmaca_运行时api模块.html#mcerror-t-mcfuncgetattribute-int-value-mcfunction-attribute-attrib-mcfunction-t-hfunc
[dynamic-limit]: https://developer.metax-tech.com/api/client/document/preview/990/split_files/mxmaca_运行时api模块.html#mcerror-t-mcfuncsetattribute-const-void-func-mcfuncattribute-attr-int-value

[cpp-shuffle]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#warp-shuffle
[cpp-reduce]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#warp-reduce
[cpp-sync]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#pddnkif8w7ir1
