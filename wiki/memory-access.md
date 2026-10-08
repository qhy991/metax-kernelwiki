# C550 contiguous copy and strided gather

[Home](../README.md) · [Catalog](../data/catalog.json) · [Probe guide](../experiments/native/README.md)

Run `20261007-native-01` checked 49 native probe cases in a fixed-order sweep. For its largest FP32 copy, the observed median event-batch mean decreased at every step from block size 64 to 512. For gather with block size fixed at 256, it increased with stride. A subsequent four-process experiment with balanced order confirmed the copy ranking at that size. Gather still has only the initial sweep, and the hardware mechanisms behind the differences remain unresolved.

The evidence scope is `local-measurement` and the knowledge label is `locally-measured`. The [result record](../data/results/20261007-native-01.json) owns the values and conditions. See the [native README](../experiments/native/README.md) for the contract and reproduction procedure. Execution used `experiments/native/probe.cpp` at commit `cbdea92d92631a5cb388916bc985a0ceab7e3f16`.

## What block size and stride change

Copy computes `i = blockIdx.x * blockDim.x + threadIdx.x` and, when `i < N`, performs `output[i] = input[i]`. Changing block size changes threads per block, total block count and inactive tail threads while preserving each valid element's read/write and the output contract. It is a candidate execution-configuration change; current resource and instruction evidence does not establish which changed factor dominates the time.

Gather uses the same thread mapping but reads `input[i * stride]` and writes consecutive `output[i]` elements. Stride is measured in FP32 elements. At fixed output count, increasing stride expands the input-index span and the spacing between neighboring threads' reads, while preserving output count and logical read/write bytes. Different strides select different input indices. Replacing a gather with stride 1 is therefore not automatically a semantics-preserving optimization. If an application permits input packing or reordering, preparation costs and the full invocation must also be compared.

Because access spacing and input span change together, these timings do not separate transaction coalescing, cache state, scheduling and other effects. Explanations without profiler metrics or disassembly remain hypotheses. These values do not identify cache-line size, bank size or DRAM traffic.

## Initial measurements

Each case performs 20 warmups and synchronization, then retains 10 samples of 100 launches. The tables show the median of each `mcEventElapsedTime` batch divided by 100, in microseconds. This summarizes a batch mean, not isolated single-kernel latency. The [result record](../data/results/20261007-native-01.json) retains every sample and condition.

Contiguous copy, `N = 4,194,317` FP32 elements:

| Threads per block | Median event-batch mean (µs) |
| ---: | ---: |
| 64 | 67.706 |
| 128 | 57.555 |
| 256 | 47.502 |
| 512 | 43.383 |

Strided gather, `N = 1,048,573` FP32 outputs, block size 256:

| Input stride (elements) | Median event-batch mean (µs) |
| ---: | ---: |
| 1 | 15.660 |
| 2 | 16.637 |
| 4 | 19.546 |
| 8 | 30.730 |
| 16 | 52.347 |

These initial rankings come from one fixed-order sweep. Alone, they do not establish a generally optimal block, a general speedup or a stable ranking. Any reported effective bandwidth counts only one logical read and one logical write per valid element: `8*N` bytes for FP32. Holes in the input span are not useful bytes. This metric is not measured DRAM bandwidth.

## Four-process confirmation with balanced order

[20261007-copy-confirm](../data/results/20261007-copy-confirm.json) used the same source, binary and `N=4,194,317` input in four fresh processes with predetermined orders: `64,128,512,256`; `128,256,64,512`; `256,512,128,64`; and `512,64,256,128`. Each block size occupies each position once, and each of the twelve directed adjacent pairs occurs once. All outputs and guards passed for the 16 cases. CPU checking ran only after each device process exited and released its allocation.

The analysis first takes the median of each process's ten batches, then reports median [min, max] across **processes**:

| Block size | Event-batch mean (µs), four-process median [min, max] |
| ---: | ---: |
| 64 | 67.632 [67.363, 68.050] |
| 128 | 58.113 [57.585, 58.184] |
| 256 | 48.548 [48.268, 49.249] |
| 512 | 44.376 [43.982, 44.460] |

All four processes produced the same ranking. The within-process `T64/T512` ratio is **1.528 [1.522, 1.532]**. This is a local ratio for this copy and timing contract. It shows that the initial trend survives the balanced orders; it does not establish a universal best block size or end-to-end operator gain. Clocks were not fixed and external jobs were not excluded. Balanced order does not eliminate all nonlinear changes in device state.

## Correctness and environmental limits

The input is `input[i] = float32(i)`. Values are unique, finite and exactly representable over the indices used. The CPU oracle checks every final output bit pattern and 32 guard words on each side. All 49 initial cases passed. Copy includes partial-block tails; gather checks the distinct indices actually selected by each stride. See the [native README](../experiments/native/README.md) for the full case list. Guards cover nearby boundary writes, not arbitrary invalid reads, distant writes or every intermediate repeated launch.

The environment was MACA SDK `3.5.3.18` and mxcc `1.0` (`6477545d4d`). The runtime reported MetaX C550, execution-group width 64, 104 multiprocessors and 8 MiB of L2. These are observations from that API version. In the inspected headers, `waveSize` and `warpSize` are aliases, not independent hardware confirmations. The probe does not use them to infer cache behavior or peak performance.

Events use the default stream. Their intervals exclude allocation, host-device transfers and file writes, but may include device idle gaps during host submission. Separately recorded host enqueue time includes `mcGetLastError` after every launch; it is not pure hardware launch overhead.

`MACA_LAUNCH_MODE`, `MACA_LAUNCH_BLOCKING` and `MACA_DIRECT_DISPATCH` were unset. Addresses repeat and the application applies no explicit cache reset. The effective runtime cache policy is unverified, so this is labeled neither a warm-cache experiment nor an unflushed-cache experiment. Execution uses a cooperative `local_serialized` device lock. External activity was not excluded, and system-wide exclusivity is not claimed. The initial sweep collected no profiler metrics, so the timing differences do not identify a unique bottleneck.

## What the next controls must distinguish

The local copy ranking has been confirmed across processes. A changed working set, tail fraction or data mapping requires new validation; this ratio does not carry over automatically. The initial gather trend still needs confirmation with balanced order and a control that fixes input span while changing thread-address spacing.

Separate traces examine actual kernel intervals and compiled-resource fields. Profiled timing remains separate from unprofiled measurements. Attribute an observation to a mechanism only when measurements distinguish competing explanations; unset launch environment variables do not establish warm caches.

## Successor with a fixed unique address set

The [read-order experiment](memory-order.md) fixes each size's unique input address set, span and logical traffic, while reusing the same pair of device buffers within a process. Six-process confirmation still found non-monotonic timing across permutations. This removes the changing whole-input span from the original sweep's confounds, but does not separate short-term reuse windows, address translation or transaction formation. The different permutations are also not semantics-preserving replacements for the original gather.
