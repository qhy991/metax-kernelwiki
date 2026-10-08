# Fixed-footprint read-order experiment

[Home](../../README.md) · [Measured findings](../../wiki/memory-order.md)

Question: for the same input address set, allocation and logical byte count,
how does changing the read order affect this C550 kernel's batch timing?
This is a separate microbenchmark. Different shifts produce different output
permutations; they are not equivalent optimizations of the earlier native gather.

## Workload and oracle

The CPU preparation writes 18 cases: `N=2^16, 2^20, 2^24`, each with
`shift=0, 2, 4, 6, 8, 12`. Block size is always 256. All lengths are exact
multiples of the block size; this plan establishes no partial-block tail coverage.
The TSV columns are `id, n, shift, block, warmups, samples, launches` (tab-separated).
An admitted plan may contain a reordered subset of these cases, including one
case for an isolated trace. Other lengths, shifts and timing settings are refused
on the host before any device API call.

All cases call one kernel with runtime parameters `n`, `shift` and `log2_n`:

```cpp
uint64_t j = ((i << shift) | (i >> (log2_n - shift))) & (n - 1);
output[i] = input[j];
```

The index uses unsigned 64-bit arithmetic, so the largest intermediate left
shift is below `2^36`; no 32-bit truncation is needed. No template or separate
kernel specializes a shift. The expression also runs for shift zero.

The independent CPU oracle views input as a row-major matrix with `N/2^shift`
rows and `2^shift` columns, transposes it, and flattens the result. Its index is
`j = (i % (N/2^shift)) * 2^shift + i // (N/2^shift)`, using integer division
and remainder instead of the GPU bit-rotation expression. For example,
`N=8, shift=2` reads indices `[0,4,1,5,2,6,3,7]`. The inverse is
`i = (j % 2^shift) * (N/2^shift) + j // 2^shift`. This bijection visits every
index from zero through `N-1` exactly once for every admitted shift. Output stores
are consecutive for all shifts.

Input element `i` is `float32(i)`, for all `0 <= i < 16,777,216`. These values
are finite, unique and exactly representable. The CPU checker verifies every
input value, every output bit pattern, finite payload counts, and 32 guard words
on each side of each final output. Guards and initially unwritten payload use
`uint32 0xffffffff`; unwritten values fail the finite and equality checks.

One 64 MiB input device allocation and one 64 MiB + 256 byte output device
allocation are reused for the whole process: peak explicit device buffers are
128 MiB + 256 bytes. Input and payload base addresses do not change between
cases. Each case resets its `N+64` output words outside timing, and saves those
words after all launches. Same-`N` cases therefore read the same address set and
write the same address set with the same footprint. Guards detect nearby writes,
not distant writes or invalid reads. Only final outputs are checked; repeated
intermediate launches have no independent snapshots.

## Procedure

Keep inputs, binaries, output and logs outside the source checkout. Commit the
source before device execution, then use that frozen checkout. Preparation,
explicit-target compilation, CPU input validation and final checking need no GPU
lease. Use the installed gpu-infra lifecycle and the node's existing allocator;
this directory introduces no runner or allocation protocol.

```sh
python3 experiments/memory_order/experiment.py prepare /tmp/metax-order-input
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  bash experiments/memory_order/compile.sh /tmp/metax-order-probe
mkdir /tmp/metax-order-output
# Inside the existing allocator, with its external process timeout:
/tmp/metax-order-probe --run /tmp/metax-order-input /tmp/metax-order-output
# Release the lease after the process exits, then run on the CPU:
python3 experiments/memory_order/experiment.py check /tmp/metax-order-input /tmp/metax-order-output
```

`xcore1000` is the inspected compiler target, distinct from the C550 device's
reported XCORE1002 identity. Compilation uses `-x maca`; it must not auto-detect
a GPU. The executable requires exactly one visible device named `MetaX C550`
and records its PCI bus ID and runtime device properties. Checked calls to
`mcRuntimeGetVersion` and `mcDriverGetVersion` retain raw integer results as
`runtime_version_api` and `driver_version_api`; no version encoding is inferred.
Its output directory
must already exist and be empty; existing results are not reused. Runtime API
errors, invalid timing, incomplete records, missing outputs and failed oracle
checks return nonzero. Freeze and retain case order for each process; the default
order alone does not control cross-process thermal or order effects.

## Timing and trace scope

Each case performs 10 warmups followed by device synchronization, then 10 event
batches of 10 launches: exactly 110 kernel launches per case. There is no extra
first-launch probe. Raw JSONL retains all `event_batch_ms` and
`host_enqueue_batch_us` values. Events use the default stream. Host enqueue time
includes per-launch error queries. Event time may contain device idle gaps while
the host submits work. Division by ten yields a batch average, not isolated
kernel latency or pure launch overhead. Allocation, memset, transfers and file
writes are outside the event intervals.

The same addresses recur. The application performs no explicit cache reset, and
effective runtime cache policy is unknown. Raw protocol records retain
`MACA_LAUNCH_MODE`, `MACA_LAUNCH_BLOCKING`, `MACA_DIRECT_DISPATCH`,
`MACA_CACHE_PATH` and `MACA_CACHE_DISABLE` as strings or null when unset. Preserve
raw local paths privately and redact them in public projections. Those settings
do not themselves prove actual cache behavior. Logical traffic is `8*N` bytes
per launch; logical GB/s is not measured DRAM traffic or bandwidth. Even with
fixed footprints, timing alone does not separate coalescing, cache locality,
address arithmetic, scheduling, clocks and host submission effects. This kernel
collects no hardware counters.

Collect profiling separately from unprofiled timings. For one admitted case,
the existing trace summarizer can use `--expected-launches 110 --warmups 10`. Do not
apply one cumulative warmup removal to a multiple-case trace: each case has its
own ten warmups. Successful profiler startup alone is not evidence of valid GPU
events. Frequency, power and concurrency limits belong in the run record.

## CPU checks

Run `python3 -m unittest discover -s tests -p test_memory_order.py`. Tests
exhaustively check small permutations and their inverse, full address coverage,
the independent transpose oracle, wrong ordering, unwritten payload, both guards,
invalid lengths/shifts, mismatched metadata and missing or invalid timing samples.
They do not establish MXCC compilation or GPU correctness. Python reuses only
the existing native probe's CPU input validation, binary reading, endian check
and input/guard constants; its permutation and experiment protocol are separate.
