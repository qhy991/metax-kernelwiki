# C550 transpose: tiling, shared pitch and capacity controls

[Home](../README.md) · [Catalog](../data/catalog.json) · [Probe guide](../experiments/transpose/README.md)

The first study compared direct access, a 64×64 shared-memory tile and a tile with one extra padding column for the same FP32 transpose. All outputs and guards passed for 57 cases. In six-process comparisons at three large shapes, the unpadded tile achieved event-interval speedup ratios of approximately **9.18×, 7.52× and 2.83×** against this direct-access baseline.

Padding had no consistent direction of benefit across the first two specialized tiled kernels. A successor fixed the declared allocation capacity and used one runtime-parameter kernel for both pitches; pitch65 was faster at all five large shapes tested. A third study separated dynamic shared-memory requests from pitch: the request response depended on shape, while the pitch effect remained at a fixed request. Each study retains its own conditions and evidence. None establishes an unconditional C550 rule. The first study's [complete samples and resources](../data/results/20261008-transpose.json) remain unchanged.

## The implementations produce the same result

The operation is out-of-place, contiguous row-major FP32 transpose:

```text
B[col * rows + row] = A[row * cols + col]
```

At a fixed shape, all three implementations read and write the same elements and produce identical outputs. Each process reuses the same pair of device buffers. The [earlier read-order experiment](memory-order.md) changed output permutations and serves only to motivate hypotheses; its timings are not the baseline here.

| Implementation | Thread block | Work and storage per block |
| --- | --- | --- |
| `direct` | 256×1 | Each thread writes one consecutive output and locates its input using division/remainder |
| `tile64` | 64×4 | Cooperatively load a 64×64 tile, synchronize, then exchange coordinates for stores; declared shared array is 64×64 |
| `tile64_pad1` | 64×4 | The same tile and synchronization structure; declared shared array is 64×65 |

Tiling jointly changes address arithmetic, block count, work per thread, shared storage and barriers. Its ratio against direct access is a **whole-implementation comparison**, not an isolated measure of coalescing. The two tiled sources differ only in shared row pitch, but that can also change generated address arithmetic and resource use.

## Why edge stores read initialized shared elements

A load writes `tile[u][v]` when:

```text
by*64 + u < rows && bx*64 + v < cols
```

A store reading `tile[tx][ty+j]` independently checks the transposed coordinates:

```text
by*64 + tx < rows && bx*64 + ty+j < cols
```

The store condition is exactly the validity condition of the earlier load for that shared element. All 256 threads reach the same `__syncthreads` without early returns. A thread whose own load is out of bounds must not skip the barrier or reuse its load guard to decide whether to store.

Coverage includes one-element and skinny matrices, `31×33`/`33×31`, all nine `63/64/65` dimension combinations, and five large matrices including ragged `262143×63` and `4095×4097`. Every final output was compared bitwise with independent CPU row/column loops, and all 32 guard words on each side remained intact. See the [transpose README](../experiments/transpose/README.md) for source and reproduction. This does not check every intermediate launch or provide a general race sanitizer.

## Six-process confirmation for matching shapes

The first study used source `bf1db0f` on C550 with MACA 3.5.3.18 and MXCC `1.0.0 (6477545d4d)`. Each case performed 10 warmups and 10 batches of 10 launches. Six processes followed predetermined shape/implementation orders, placing each implementation at each position twice within a shape. Full CPU checking followed process exit and device release.

Times below first take each process's batch median, then report six-process median [min, max], in µs:

| Input shape | direct | tile64 | tile64_pad1 |
| --- | ---: | ---: | ---: |
| 262144×64 | 1555.738 [1555.469, 1556.275] | 169.459 [168.986, 169.894] | 169.651 [169.357, 170.278] |
| 65536×256 | 1274.528 [1270.515, 1279.706] | 169.517 [169.114, 169.779] | 171.283 [170.893, 171.917] |
| 4096×4096 | 475.795 [475.187, 476.096] | 167.968 [167.501, 168.384] | 165.734 [165.504, 166.323] |

Ratios are calculated **within the same process and shape**, then summarized. They are not ratios of aggregate medians:

| Shape | T(direct)/T(tile64) | T(tile64)/T(pad1) |
| --- | ---: | ---: |
| 262144×64 | 9.183 [9.156, 9.205] | 0.999 [0.995, 1.003] |
| 65536×256 | 7.523 [7.498, 7.548] | 0.991 [0.984, 0.991] |
| 4096×4096 | 2.833 [2.826, 2.842] | 1.012 [1.011, 1.016] |

The second column measures benefit against this direct-access baseline. In the third column, a value above 1 means padding is faster; below 1 means slower. These three shapes do not establish a generally optimal tile or dispatcher rule. At this stage, the two large ragged shapes had full correctness checks and exploratory timing, but no six-process performance confirmation.

## Resource reports are not bank-conflict measurements

Separate traces covered all three implementations at `4096×4096`. Each contained 110 actual kernel events and passed final-output checking. Function queries and traces agreed on these reports:

| Implementation | Reported registers per thread | Reported static shared bytes |
| --- | ---: | ---: |
| direct | 14 | 0 |
| tile64 | 13 | 16,384 |
| tile64_pad1 | 13 | 16,640 |

Reported shared use is retained separately from the source array's intended size, not replaced with it. The device API also reports 65,536 bytes of shared memory per multiprocessor. These values describe resource constraints, not actual occupancy. No validated bank-conflict, cache or DRAM-traffic counters were collected.

The [official tuning guide's C500 chapter][c500-banks] describes 32 banks and phased access by a 64-thread warp. It does not establish this C550's bank contract. A padding-related timing change alone does not identify bank count, mapping or conflict degree. Tiling also changes global read/write organization, per-thread work, synchronization and resources.

This study uses one position-distinguishing finite FP32 input. It does not qualify in-place transpose, arbitrary strides, other dtypes, other SDKs/GPUs or target-framework integration. Event intervals may include host submission gaps; data-cache policy is uncalibrated and clocks are not fixed. Allocation is cooperative `local_serialized`. Profiled timing remains separate from ordinary measurements, and raw trace time units remain unverified.

<a id="后继同一个函数固定16640字节容量"></a>

## Successor: one function with a fixed 16,640-byte capacity

Source `56db27e` adds `runtime_pitch64` and `runtime_pitch65` controls for the original 19 shapes. Both use the non-template `transpose_runtime_pitch_kernel` and declare `__shared__ float storage[64*65]`. Only runtime pitch changes between 64 and 65. Global reads/writes, block/grid geometry, barrier, input and CPU oracle stay the same.

```text
load:  storage[(ty+j)*pitch + tx]
store: storage[tx*pitch + ty+j]
```

Pitch64 uses the first 4096 slots. Pitch65 uses 4096 slots and reaches index 4158. Both fit the 4160-element capacity. Source construction fixes declared capacity, not the active shared-address set or span. Function queries and two separate traces additionally report 13 registers and 16,640 static shared bytes for both arguments. The trace function name is the same, with 110 kernel events per capture. These reports do not measure actual occupancy.

After all 38 parameter cases passed, ten independent processes confirmed three regular and two ragged large matrices. Each shape appears at each shape position twice, once with each pitch order. This is not a claim of complete cross-case carryover balance. All 100 confirmation cases passed.

The table computes `T64/T65` from within-process batch medians and then reports ten-process median [min, max]. A value above 1 means pitch65 is faster:

| Shape | Pitch64 time (µs, median) | Pitch65 time (µs, median) | T64/T65 |
| --- | ---: | ---: | ---: |
| 262144×64 | 199.379 | 170.099 | 1.173 [1.167, 1.174] |
| 65536×256 | 200.864 | 172.051 | 1.168 [1.158, 1.171] |
| 4096×4096 | 197.568 | 166.202 | 1.189 [1.186, 1.191] |
| 262143×63 | 202.534 | 183.981 | 1.101 [1.098, 1.105] |
| 4095×4097 | 256.883 | 238.048 | 1.079 [1.075, 1.082] |

All ten processes observed the same direction for each shape. The [1400 raw timing batches, full-output check summaries and traces](../data/results/20261008-shared-pitch.json) are retained separately and do not overwrite the first study.

The result supports an effect of shared pitch on complete-kernel time for this common function and its reported resources. The old and new experiments change both capacity conditions and code generation. They cannot attribute the first study's effects entirely to capacity or resident-block count, nor identify C550 bank geometry. Absolute times across the two implementations do not isolate a cause.

The next control uses the [official dynamic-shared interface](../docs/toolchain.md#dynamic-shared-memory-capacity-as-a-follow-up-control) to vary the capacity request more narrowly.

<a id="再后继同一函数的动态shared请求量"></a>

## Further control: dynamic shared-memory requests in one function

Source `dd20525` changes storage to `extern __shared__ float storage[]` and compares three valid configurations in one non-template kernel: A is pitch64/request16,384B; B is pitch64/request16,640B; C is pitch65/request16,640B. A/B keep kernel arguments, shared addresses, global accesses and launch geometry the same, changing only the dynamic-shared launch request. B/C compare pitches at a fixed request. Pitch65/16,384B violates this suite's capacity contract and is refused on the host without an invalid GPU access.

All 57 cases covering the original 19 shapes passed. Confirmation used ten predetermined processes, with each large shape following `A_first,B,C,A_last` or `A_first,C,B,A_last`. Five cyclic shape orders balance shape positions and the middle B/C order. Each case first takes the median of ten batch means; then `T_A=(T_A_first+T_A_last)/2`. Ratios are computed per process and summarized across ten processes.

| Shape | A (µs) | B (µs) | C (µs) | T_A/T_B, median [min,max] | T_B/T_C, median [min,max] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 262144×64 | 169.174 | 199.168 | 170.330 | 0.849 [0.849, 0.851] | 1.169 [1.166, 1.174] |
| 65536×256 | 169.507 | 200.614 | 172.166 | 0.845 [0.843, 0.847] | 1.167 [1.154, 1.169] |
| 4096×4096 | 168.138 | 197.421 | 166.541 | 0.851 [0.848, 0.854] | 1.185 [1.180, 1.190] |
| 262143×63 | 179.514 | 202.483 | 184.422 | 0.887 [0.885, 0.888] | 1.098 [1.094, 1.104] |
| 4095×4097 | 257.619 | 256.435 | 240.851 | 1.005 [0.999, 1.009] | 1.064 [1.062, 1.070] |

The A/B/C time columns are ten-process medians. Ratios are computed within each process, not from those table aggregates. A/B below 1 means the smaller request A is faster; B/C above 1 means pitch65 C is faster at the same request. All ten processes observed A faster than B for the first four shapes. For `4095×4097`, A/B is 1.0047 [0.9994,1.0092], so the direction is not consistent. B/C has a consistent direction at all five shapes. Dynamic-shared requests can affect this kernel's time, but the response depends on shape; a pitch effect also remains when the request is fixed.

Matching endpoint configurations monitor drift. The 50 `T_A_first/T_A_last` ratios span approximately 0.9939–1.0029. Every endpoint and each endpoint's ratio to B is retained. This is not complete carryover control. No process was excluded for drift or speed, and the last shape's small difference is not reported as a stable capacity advantage.

Three independent traces contain 110 events with the same kernel name each. Function queries report 13 registers, zero static shared memory and a dynamic limit of 65,536B. Trace dynamic-shared fields report 16,384/16,640/16,640B respectively. Each trace has 110 false recompilation flags with no missing values; its fresh binary-cache directory remained empty. Launch requests, API limits and tool descriptors are separate observations. They prove neither rounded physical allocation nor identical final machine code. Dividing `65536 / requested_bytes` cannot establish a reduction from 4 to 3 resident blocks.

The [260 cases, 2600 raw timing batches and check summaries](../data/results/20261008-dynamic-shared.json) comprise 57 sweep cases, 200 confirmation cases and 3 trace cases. All 3,646,275,267 payload-element checks and 16,640 guard-word checks passed. These are repeated checks, not that many distinct random inputs. All 14 device processes and 3 profiled applications exited and passed release observations.

The study supports separating pitch access from launch-capacity requests experimentally. It does not estimate their full interaction or attribute the absolute difference between static and dynamic source implementations to one mechanism. Defined residency or performance-counter evidence is needed to distinguish resource scheduling from runtime paths. A subsequent independent experiment has checked the [bounded semantics of 64-lane collectives](wave-collectives.md).

[c500-banks]: https://gitee.com/metax-maca/mxmaca-performance-tuning-guide/blob/65a3f7680ec6236a8be4a24a40f830eb63218ee7/guide/ch3.Kernel编程入门.reduction.md
