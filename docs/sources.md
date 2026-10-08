# Primary source index

[Home](../README.md) · [Toolchain](toolchain.md) · [Methodology](methodology.md)

Accessed **2026-10-07**, with additional C++ intrinsic and LLVM artifact-format checks on **2026-10-08**. This page indexes sources; it is not a C550 measurement record. Device evidence must retain the product, native architecture, software versions, commands, and raw outputs, with a link to the relevant experiment.

## Product and version scope

MetaX's **MXC500 series** is a software-documentation support group containing C500, C500X, C550, C550-PL, C588, N260, and X206. Membership does not imply equal capacity, bandwidth, caches, or compute capability. Official release notes separately identify MXC600 and MXC600-U. Keep C500 microarchitecture tutorials labeled as C500 material until a C550 experiment supports a target-specific conclusion. [Release overview][release-overview]

This search found no public official product specification sufficient on its own to establish the **C550 → xcore1002** mapping. That mapping, memory capacity, processor count, and native ISA need evidence from the actual C550 device queries and compilation artifacts. They cannot be inferred from C500 documentation or CUDA compatibility values.

## Checked documentation entry points

Runtime-guide links now select the official **3.5.3.x** version. The [legacy `/preview/567` entry][runtime-legacy] remains for provenance: its active version selector read **3.0.0.x** on 2026-10-07, so it does not identify the installed host's 3.5.3.18 version. The execution-model, device-query, and compilation chapters at the 3.5.3.x links were checked separately. Series documentation still does not establish local C550 validation.

| Source | What it supports | Boundary |
| --- | --- | --- |
| [Runtime API Programming Guide 3.5.3.x][runtime] | The execution model calls 64 threads a wave; entry points for device queries, memory, events, kernel launches, and MXCC builds. | Series-level guidance; query C550 for device values. Examples use `mcDeviceProp_t.waveSize`, while installed SDK headers may contain compatibility aliases. |
| [Runtime Guide 3.5.3.x: Binary Cache][runtime-cache] and [environment variables][runtime-env] | Section 4.3 describes recompilation and binary caching when the unannotated vector-add example uses a 1024-thread block, exceeding 512. Section 4.4 defines the cache path and disable switch. | This documents that example, not a performance guarantee for every kernel. Distinguish device limits, function attributes, first launch, and later execution locally. |
| [MXMACA C++ language extensions 3.5.3.x][cpp-extensions] | Shuffle mask, width, and source-lane rules; integer collectives and synchronization semantics. | The active version selector was checked. The series contract still needs installed-header, compilation, and C550 correctness checks. See [toolchain caveats](toolchain.md#64-lane-collective-contracts-and-measured-scope) for ambiguous wording. |
| [MXMACA release overview][release-overview] | Software-component versions and supported product families. | `latest` is mutable and does not identify installed software. |
| [MXMACA new features and changes][release-changes] | The SDK 3.5.3.18 section lists `-offload-arch=native`; its stated product group includes C550. | The installed compiler still needs its own option check. Other sections have different product scopes; their features do not automatically apply to C550. |
| [MXMACA known issues and limitations][release-limits] | Version- and scenario-specific leads, including some communication-operator issues on C550 OAM. | Do not generalize a scoped limitation to all C550 kernels or copy environment settings as optimization defaults. |
| [Official performance-tuning guide README][guide] | Navigation for `guide/`, `case/`, and `microbenchmark/`; the repository names C500 and A100 as its test devices. | C500 measurements are hypotheses to test locally. No license was declared on the inspected page; prefer links, paraphrases, and independently written probes. |
| [Official vector-add tutorial][vector-guide] | Examples of the `cucc` route, device memory, warmup, event timing, and error checking. | The tutorial targets C500. CUDA-named interfaces are compatibility APIs, not NVIDIA hardware identity. |
| [mcProfiler manual PDF][profiler] | Entry point for performance-counter collection, task parameters, and metric selection. | This is an older manual. The installed C550 tool's CLI, permissions, and metric set have not been validated in this work. |
| [mcTracer manual 3.5.3.x][tracer] and [Viewer chapter][tracer-viewer] | Collection options, JSON outputs, and the dedicated Viewer. The `/preview/995` version selector also identifies 3.5.3.x. | The collection and Viewer chapters were read, but no exported JSON `ts`/`dur` unit contract was found. Upstream MCPTI timestamp units or numerical scale cannot substitute for exporter evidence. |
| [mxvs test-suite contents 3.5.3.x][mxvs] | Official device-information, PCIe, Memory, MetaXLink, and compute tests. | Their timing scope differs from a custom kernel's and needs separate reporting. This work has not run them. |

## Checked official code

The mcTriton links below are pinned to **`7dd407c26568fceaca44cb894138e5202d369805`**, on the `3.0` branch. These are observations of published source, not proof of the installed C550 wheel's source identity.

| File | Implementation available for inspection |
| --- | --- |
| [mcTriton README][triton-readme] | Build dependencies from the MACA stack and `metax_llvm`; build script `maca_tools/build_triton.sh`. |
| [backend/driver.py][triton-driver] | Returns the `maca` backend and a 64-lane target; the launcher uses `64 * num_warps` threads. |
| [backend/driver.c][triton-driver-c] | Maps device `major` to Triton capability: 10→80, 15→86, 16→89. Resource queries use MACA APIs. |
| [backend/compiler.py][triton-compiler] | `mcfatbin` output, compilation stages, MACA pipeline options, and `num_warps` checks. |
| [triton_metax.cc][triton-codegen] | Forms a fatbin from LLVM IR through MXCC; includes switches to print the compilation command and retain intermediate files. |
| [vLLM-metax event-timing wrapper][kernel-timer] | Pinned at `f2fcc59c314f1fbc7897f46d465e5f7c35900e8a`; an implementation using `mc_runtime.h` and `mcEventElapsedTime` directly. |
| [mcTVM][mctvm] | Another official compilation route. Its README's `metax/mxc-c500` label explicitly targets C500. |

## Compiled artifact formats

The [compiled-artifact guide](compiled-artifacts.md) links LLVM 19.1 documentation for binary bundles, bitcode wrappers, section extraction, bitcode decoding and contraction flags. These references explain format and tool contracts. The local inspection separately verifies the installed MetaX tools and the contents of the retained q7 executable. Upstream LLVM documentation does not specify MetaX runtime payload selection or WMMA precision.

## Shared memory and synchronization

The [Runtime 3.5.3.x WSM description][wsm-scope] scopes shared storage to a thread block. The [collective-operation section][block-sync] requires participating threads to match synchronization calls and relates block synchronization to `__syncthreads`. The tiled transpose in this wiki therefore masks boundary loads and stores separately while letting the whole block reach the same barrier.

The [pinned chapter 3 of the official tuning guide][c500-banks] explicitly describes **C500**: 32 banks, consecutive 4-byte units, and 32-bit accesses by a 64-thread warp split into two 32-thread phases. That passage does not name an SDK version. This search found no equivalent C550-specific bank contract. The passage supports predictions to test on C550, not a C550 hardware constant. Even a benefit from one extra padding column cannot identify the bank count or conflict degree.

## Initial falsifiable questions

These are experiment questions, not established hardware conclusions. Later local findings are linked from the [research log](research-log.md).

1. **Identity and execution groups:** Do C550 device queries, the native compiler target, Triton target, and actual block size agree? Record native architecture and compatibility capability separately; check 32-, 64-, and 128-thread boundaries and tail correctness.
2. **Timing scope:** How do single-launch events, amortized event batches, and host synchronization timing differ for the same kernel? Use empty kernels and controlled workloads first, then report launch cost, batch means, and measurement noise separately.
3. **Memory access:** Do sequential, strided, misaligned, and vectorized accesses differ consistently? Hold logical read bytes and correctness fixed; record emitted instructions and cache state. Effective bandwidth is not automatically HBM bandwidth.
4. **Capacity and reuse:** Does a working-set sweep show reproducible latency or bandwidth transitions? Separate repeated reuse, independent addresses, and explicit resets. One transition cannot identify a cache level or its capacity.
5. **WSM access:** How do shared-memory stride, broadcast, and padding affect latency? Measure access patterns before interpreting bank structure; do not assume NVIDIA's bank count or width.
6. **Compiler decisions:** For a validated computation, do `num_warps`, `num_stages`, or `basic/cpasync` change the artifact, correctness, or timing? Admit only options supported by the installed version; an option name does not establish a hardware instruction.

[runtime]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/index.html
[runtime-legacy]: https://developer.metax-tech.com/api/client/document/preview/567/C500_RuntimeAPIProgrammingGuide_CN.html
[runtime-cache]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编译和调试.html#binary-cache
[runtime-env]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编译和调试.html#irs9wigbg1oh1
[release-overview]: https://developer.metax-tech.com/api/client/document/preview/发布说明/MXMACA_发布说明/曦云C500系列/latest/split_files/概述.html
[release-changes]: https://developer.metax-tech.com/api/client/document/preview/发布说明/MXMACA_发布说明/曦云C500系列/latest/split_files/新增特性及变更.html
[release-limits]: https://developer.metax-tech.com/api/client/document/preview/发布说明/MXMACA_发布说明/曦云C500系列/latest/split_files/已知问题和使用限制.html
[guide]: https://gitee.com/metax-maca/mxmaca-performance-tuning-guide/blob/main/README.md
[vector-guide]: https://gitee.com/metax-maca/mxmaca-performance-tuning-guide/blob/main/guide/ch1.初探异构编程.vectoradd.md
[profiler]: https://developer.metax-tech.com/api/client/document/file/211/preview/?file_type=pdf
[tracer]: https://developer.metax-tech.com/api/client/document/preview/性能测试及分析工具/mcTracer使用手册/曦云C500系列/3.5.3.x/split_files/mctracer.html
[tracer-viewer]: https://developer.metax-tech.com/api/client/document/preview/性能测试及分析工具/mcTracer使用手册/曦云C500系列/3.5.3.x/split_files/mctracer_viewer.html
[mxvs]: https://developer.metax-tech.com/api/client/document/preview/996/index.html
[triton-readme]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/README.md
[triton-driver]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.py
[triton-driver-c]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.c
[triton-compiler]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/compiler.py
[triton-codegen]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/triton_metax.cc
[kernel-timer]: https://github.com/MetaX-MACA/vLLM-metax/blob/f2fcc59c314f1fbc7897f46d465e5f7c35900e8a/csrc/libtorch_stable/quantization/awq/hgemv_selector.hpp
[mctvm]: https://github.com/MetaX-MACA/mcTVM

[wsm-scope]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编程模型.html#3s9wt8x546le1
[block-sync]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编程接口.html#f9dqemikxd6i1
[c500-banks]: https://gitee.com/metax-maca/mxmaca-performance-tuning-guide/blob/65a3f7680ec6236a8be4a24a40f830eb63218ee7/guide/ch3.Kernel编程入门.reduction.md

[cpp-extensions]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html
