# C550 launch bounds: device limits, function attributes and recompilation

[Home](../README.md) · [Catalog](../data/catalog.json) · [Probe guide](../experiments/native/README.md)

In this C550 / MACA 3.5.3.18 experiment, a reported function attribute of `maxThreadsPerBlock=512` **did not prevent correct execution with a 1024-thread block**. The default copy used a runtime variant carrying the recompilation flag. With an explicit `__launch_bounds__(1024)` declaration on the same function, the function attribute became 1024 and a separate trace no longer showed that flagged path.

This conclusion applies to the tested kernel, MXCC version and conditions. It does not transfer CUDA's full launch-bounds semantics, a trace field or this startup timing to other kernels.

## Distinguish the reported limits

| Observation | Value in this experiment | What it describes |
| --- | ---: | --- |
| `mcDeviceProp_t.maxThreadsPerBlock` | 1024 | The device block limit declared by the runtime API |
| `mcFuncGetAttributes` for the default copy | 512, even after a 1024-thread launch | The function API's default compiled configuration; it was not a hard execution limit here |
| `max_block_size` in the default copy's 1024-thread trace | 1024, reported in 949 events and missing in 72 | A tool resource field for that execution variant; missing values remain missing |
| Function attribute with explicit `__launch_bounds__(1024)` | 1024 | The installed compiler accepted and reflected this single-argument declaration |

The installed SDK's `mcr/mc_runtime_api.h` documentation for `mcFuncGetAttributes` distinguishes the default 512 without launch bounds from hardware support for 1024. The [official 3.5.3.x runtime guide's binary-cache example][runtime-cache] also describes runtime recompilation triggered by a 1024-thread launch. These sources motivate a testable expectation; the following runs test it for this copy.

## Boundary sweep with the default function

The [18-case boundary experiment](../data/results/20261007-block1024-boundary.json), at source `088783d`, covered blocks of 512/1024 threads, one-element inputs, lengths below/equal to/above a block, an approximately 16 MiB copy, and two empty-kernel cases. All **8,405,022 payload elements and 1,024 guard words** in the 16 cases with outputs passed.

Each case first recorded one host interval from launch through completion synchronization, then performed the original 20 warmups and 10×100 timed launches. For the large copy, the median event-batch mean was 45.403/40.180 µs with blocks 512/1024. This was one fixed-order sweep, not evidence of a generally optimal block size. The first encounter with the 1024-thread copy took approximately 162 ms in the host completion interval. That interval includes any loading, compilation, submission and synchronization triggered there; it is not isolated JIT compilation time.

Separate single-case traces then used fresh processes and empty run-local binary-cache directories:

| Default copy block | Runtime function attribute | Trace `max_block_size` | Reported `is_recompiled` values | New cache files |
| ---: | ---: | ---: | --- | ---: |
| 512 | 512 | 512 | 949 false; 72 missing | 0 |
| 1024 | 512 | 1024 | 949 true; 72 missing | 2 |

Each trace contains 1021 actual GPU kernel events: one initial diagnostic launch, 20 warmups and 1000 timed launches. **949 true flags do not mean 949 compilations.** The flag belongs to a kernel-event descriptor. Two cache files are not a compilation count either.

## Control that changes one declaration

Source `bc9f748` provides two builds of the same copy body. Macro value 0 leaves it unannotated; value 1024 adds only this declaration to the copy:

```cpp
__global__ __launch_bounds__(1024)
void copy_kernel(const float* input, float* output, uint64_t n);
```

This is a declaration sketch. See [probe.cpp](../experiments/native/probe.cpp) for the implementation and the [native README](../experiments/native/README.md) for commands. Both builds use the same commit, compiler, target, block=1024, inputs and oracle. Each runs in a fresh process with an empty binary-cache directory.

| Build | Runtime function attribute | Trace `max_block_size` | Reported recompilation flags | New cache files | First host completion interval |
| --- | ---: | ---: | --- | ---: | ---: |
| Default declaration | 512 | 1024 | 949 true; 72 missing | 2 | 164.734 ms |
| Explicit bound=1024 | 1024 | 1024 | 949 false; 72 missing | 0 | 2.659 ms |

Both independently traced runs produced bitwise-correct final outputs. The explicit-bound build also passed the full 18-case boundary suite. The unchanged empty kernel still reported a function attribute of 512. The [raw samples, attributes and coverage](../data/results/20261007-launch-bound-control.json) retain these distinctions.

The control supports this conclusion: **for this copy, declaring a function bound of 1024 ahead of execution avoids the observed runtime recompilation path.** There was only one fresh-process trace per build. The first-completion times are neither a replicated speedup estimate nor pure kernel latency. Absence of new files or flags does not prove that no compiler work occurred anywhere.

## When to try it and what remains untested

If a function must launch with a block larger than its default compiled constraint, an explicit bound is a candidate for reducing first-launch work. Check the installed SDK interface, then validate complete outputs, function resources, runtime variants and steady-state performance. The declaration can also change register allocation and other code-generation decisions. This example does not decide the choice for other functions or justify always using 1024-thread blocks.

The experiment did not test the second `__launch_bounds__` argument, launches above an explicit bound, other SDKs/GPUs or end-to-end gains for a larger operator. The binary shader cache is distinct from L2/DRAM data caches; the latter were not controlled, and no memory-traffic counters were collected. Exported trace `ts/dur` units remain independently unverified. Host times above come from a C++ monotonic clock, and event times from `mcEventElapsedTime`; raw trace values were not converted to microseconds.

[runtime-cache]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编译和调试.html#binary-cache
