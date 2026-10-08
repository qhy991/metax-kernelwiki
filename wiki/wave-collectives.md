# C550 wave64: shuffle, reduction and mask-type boundaries

[Home](../README.md) · [Catalog](../data/catalog.json) · [Probe guide](../experiments/wave_collectives/README.md)

On one C550 with MACA 3.5.3.18 and MXCC `1.0.0 (6477545d4d)`, direct shuffle and signed-int reduction on complete 64/128-thread blocks passed the initial boundary checks. Source `d01ef59` ran 20 cases in forward order, 20 in reverse order and two single-case traces. All 39,744 output values and 2,688 guard words passed. The [complete result](../data/results/20261008-wave-collectives.json) contains actual outputs, 420 event batches and all check summaries.

A successor tested two full, explicitly typed mask interfaces in the same SDK. The physical wave remains 64 lanes, while the 32-bit compatibility interface reduces groups of 32 elements; see the [successor section](#successor-full-typed-masks-select-different-reduction-interfaces).

Keep three quantities separate: the physical wave has 64 lanes; shuffle width can divide it into smaller subgroups; and logical data length n can be smaller than the number of participating threads. Changing either of the latter quantities does not change physical wave width.

## How one input distinguishes the boundaries

The input contains 128 int32 values, `1..128`. Every physical thread executes the same collectives with the explicit 64-bit `unsigned long` mask `0xffffffffffffffffUL`. Each thread retains nine results: sources 0/31/32/63 at width64, the same four sources at width32, and one `__reduce_add_sync` result. Compile-time assertions check mask and int widths. Execution requires C550, exactly one visible device and a runtime-reported wave64.

The table comes from actual results for a complete 128-thread block with n=128. Every thread in each listed range receives the corresponding value:

| Operation | Threads 0–31 | Threads 32–63 | Threads 64–95 | Threads 96–127 |
| --- | ---: | ---: | ---: | ---: |
| Shuffle width64, source0 | 1 | 1 | 65 | 65 |
| Shuffle width64, source31 | 32 | 32 | 96 | 96 |
| Shuffle width64, source32 | 33 | 33 | 97 | 97 |
| Shuffle width64, source63 | 64 | 64 | 128 | 128 |
| Shuffle width32, source0 or 32 | 1 | 33 | 65 | 97 |
| Shuffle width32, source31 or 63 | 32 | 64 | 96 | 128 |
| Full-mask integer sum | 2080 | 2080 | 6176 | 6176 |

The [official 3.5.3.x shuffle contract][shuffle] selects a direct source within the width-sized subgroup using `source % width`. Thus width32/source32 selects element 0 of the caller's own subgroup, not lane32 of the whole wave. The upper subgroup has its own source, and the second physical wave does not read the first wave's input.

Reduction operates on each complete wave. An incorrect 32-lane grouping would return 528, 1552, 2576 and 3600; an incorrect block-wide reduction would return 8256. CPU negative controls reject both. In the full-input case above, every participating lane was checked for its wave's 2080 or 6176 result, not just lane0.

## Logical tails use zero inputs without removing participants

Thread i reads its input when `i<n` and otherwise supplies 0. It still executes every collective and writes nine results. Even when n=0, every physical thread remains active; an empty logical input does not permit an early exit.

For a 128-thread block with n=65, only thread64 in the second wave supplies the value 65; all other inputs in that wave are zero. Thread127 supplies zero but still receives reduction result 65. Its width64/source0 result is also 65, while all broadcasts in its upper width32 subgroup are zero. These outputs are checked. Checking only the first n threads would miss this distinction.

| Physical block size | Logical n |
| --- | --- |
| 64 | 0, 1, 31, 32, 33, 63, 64 |
| 128 | The same seven values, plus 65, 95, 96, 97, 127, 128 |

The independent CPU oracle uses input-list slices, subgroup indexing and integer sums, not the GPU's lane-bit calculations or reduction implementation. It checks all nine int32 results for every physical thread and 32 guards on each side of the output allocation, bit for bit. The unwritten sentinel `0xffffffff` differs from every expected value, including zero. Reverse-order execution repeats the same semantics in a fresh process; it is not a performance treatment or an independent random-input experiment.

## What the interface, compilation and traces establish

The [official integer-reduction contract][reduce] defines mask participation and return semantics. The installed SDK's `mxgpu_llvm/lib/clang/19/include/__clang_maca_device_functions.h` contains the integer 64-bit-mask interface and delegates to a helper with permutation and accumulation loops. Native compilation with `-x maca -offload-arch=xcore1000` and device outputs validate the bounded cases above. The API name does not guarantee one hardware instruction, and header source does not replace code-generation evidence.

Two independent traces cover block64/n64 and block128/n128. Each has 110 events with the same kernel name and reports block.x of 64/128 respectively, 16 registers, zero static/dynamic shared memory and zero private memory per thread. Function queries also report 16 registers and zero shared/local memory. Each trace has 110 false recompilation flags with no missing values. The four runs' fresh binary-cache directories remained empty. These are API/tool observations, not physical residency, instruction-count or hardware-peak measurements.

Retained event-batch means describe the complete nine-channel kernel and can include host submission gaps. Dividing them by nine does not yield an intrinsic's latency. Trace time units remain independently unverified. The experiment establishes no throughput ranking, algorithmic speedup or framework-level gain.

## Reproduction and coverage limits

The [probe and commands](../experiments/wave_collectives/README.md) define inputs, masks, output layout, timing and rejection rules. Result JSON field `output_words_uint32` is decoded directly from retained device .i32 files, including guards. It can be checked with the once-published `input_words_uint32`; oracle-generated values must not be presented as device outputs. All four device workers and both profiled applications exited and passed release observations. Allocation remains cooperative `local_serialized`.

The initial experiment covers only position-encoded nonnegative int32 input, zero padding, complete physical waves and a full 64-bit mask. It does not qualify sparse masks, inactive source lanes, partial physical waves, negative values/overflow, floating-point reductions, down/xor/vote or memory ordering. Register shuffle/reduction does not replace the [separately documented synchronization and memory-order guarantees][sync].

The successor below uses a separate contract for another full-mask interface and preserves the initial results. Neither study promoted a change to the open-cake-ir Compiler, Target or calibration.

<a id="后继两种完整typed-mask选择不同归约入口"></a>

## Successor: full typed masks select different reduction interfaces

Source `715e877` adds the explicit `C550_WAVE_MASK_TYPES=1` build mode. Every physical thread in one kernel stores two results:

```cpp
const unsigned long mask64 = 0xffffffffffffffffUL;
const unsigned mask32 = 0xffffffffU;
output[thread * 2 + 0] = __reduce_add_sync(mask64, value);
output[thread * 2 + 1] = __reduce_add_sync(mask32, value);
```

The installed header provides distinct overloads for `uint64_t` and `unsigned`. Its 32-bit helper shifts the mask into the caller's half-wave and iterates only through `MACA_HALF_WARP_SIZE`. Compile-time assertions confirm that this constant is 32, `unsigned` is 32 bits, `unsigned long` is 64 bits and exactly the same type as `uint64_t`, and both literal types match. This is evidence for the installed SDK source and build, not a cross-version hardware guarantee.

Each call uses its interface's full mask, with all 64/128 physical threads participating. The experiment does not convert a low-32-bit value to a 64-bit mask and pass it to a complete wave. Both the mask type and full-mask numeric value differ. This compares two API contracts, not a type-only intervention with a fixed numeric value.

Actual outputs for a 128-thread block with n=128 are below. Every thread in each range passed checking:

| Thread range | Full 64-bit unsigned long interface | Full 32-bit unsigned compatibility interface |
| --- | ---: | ---: |
| 0–31 | 2080 | 528 |
| 32–63 | 2080 | 1552 |
| 64–95 | 6176 | 2576 |
| 96–127 | 6176 | 3600 |

With n=65, thread127's two results are `[65, 0]`: the complete wave includes thread64's value 65, while its upper 32-element group contains only padded zeros. The runtime still reports physical wave64. Each channel's `group_width` declares an expected API semantic group that the outputs test; it does not indicate a changed device wave width.

For integer reductions in this installation, preserve the mask argument type and check the intended grouping with an oracle. The tested 64-bit interface provides the full 64-element result under this contract; interpret the compatibility interface as 32-element groups. The inspected declarations show that first normalizing masks into a `uint64_t` variable changes overload selection. That is not merely formatting. This study did not execute the partial 64-bit-mask call produced by widening a low-32-bit value.

The [separate result record](../data/results/20261008-mask-overloads.json) contains 20 forward cases, 20 reverse cases and two single-case traces in the new mode: all 42 cases, 8,832 payload words, 2,688 guards and 420 event batches passed. A mode0 binary from the same source also passed the original nine-channel 20-case regression. Combined coverage is 62 cases, 27,840 payload words, 3,968 guards and 620 batches; actual outputs are public. Default preparation and the old oracle remain compatible, while unknown or mixed suites are rejected.

The two new-mode traces contain 110 `wave_mask_types_kernel` events each, with blocks64/128 respectively. Both report 16 registers, zero shared/private memory and 110 false recompilation flags with no missing values. Five device workers and two profiled applications exited and passed release observations. Both channels are timed together and request different group operations; there is no single-channel latency measurement or claim that the 32-bit interface is faster than the 64-bit interface.

Coverage remains limited to complete physical waves, two fixed full typed masks, position-encoded nonnegative int32 input and zero-padding boundaries. Sparse masks, zero masks, early exits, negative values/overflow, floating point and other collectives need separate contracts and validation. Matrix-route work must first establish the actual installed backend and compilation target.

[shuffle]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#warp-shuffle
[reduce]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#warp-reduce
[sync]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#pddnkif8w7ir1
