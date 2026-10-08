# Complete wave64 integer collectives

Question: on the measured C550 installation, do direct-index shuffle and exact
integer reduction preserve the documented subgroup boundaries at lanes 31/32/63
and across two complete physical wave64 groups? The retained
[toolchain source notes](../../docs/toolchain.md) link the MACA C++ 3.5.3.x
direct-shuffle contract, including source-index wrapping within each width-sized
subgroup, and the signed integer reduction interface. These documentation and
source declarations require compilation and device evidence before becoming
local correctness observations.

## Fixed cases and participation

Every launch has exactly one block, with 64 or 128 physical threads. The
executable requires exactly one visible device named `MetaX C550` and an observed
runtime wave size of 64. It does not translate this experiment to another wave
width. The default 20-case plan contains:

- block 64: logical `n = 0, 1, 31, 32, 33, 63, 64`;
- block 128: the same seven lengths, plus `65, 95, 96, 97, 127, 128`.

The prepared input contains 128 little-endian int32 words, exactly `1..128`.
Physical thread `i` uses `input[i]` when `i < n`, and zero otherwise. No thread
returns early: every physical thread executes every collective and stores all
nine outputs. In particular, logical `n=0` still launches and checks complete
physical waves. A zero-padded thread can receive a nonzero broadcast or wave sum;
its output is not automatically zero.

The kernel supplies the literal full mask `0xffffffffffffffffUL` with type
`unsigned long`. A compile-time assertion requires that type to have exactly
64 bits. Another assertion requires a 32-bit signed `int` payload. Both the
protocol and case records retain mask width and hexadecimal value. Negative
tests alter CPU records or saved outputs; no undefined truncated-mask or
early-exit collective is sent to the GPU.

The TSV columns are `id, block, n, warmups, samples, launches` (tab-separated).
An explicitly saved reordered subset is allowed. Invalid block/length pairs,
timing settings, IDs or input words are refused on the host before any device
API call. There is no arbitrary mask or subgroup-width launcher option.

## Nine-channel output contract

Output is thread-major: `output[thread * 9 + channel]`.

| Channel | Operation | Width | Direct source |
|---:|---|---:|---:|
| 0 | `__shfl_sync` | 64 | 0 |
| 1 | `__shfl_sync` | 64 | 31 |
| 2 | `__shfl_sync` | 64 | 32 |
| 3 | `__shfl_sync` | 64 | 63 |
| 4 | `__shfl_sync` | 32 | 0 |
| 5 | `__shfl_sync` | 32 | 31 |
| 6 | `__shfl_sync` | 32 | 32 |
| 7 | `__shfl_sync` | 32 | 63 |
| 8 | `__reduce_add_sync` | Full physical wave64 | — |

Width-32 source 32 selects index 0 of the caller's own 32-element subgroup;
source 63 selects its index 31. A caller in the upper half of a wave therefore
does not receive a value from its lower half. The reduction result is returned
to all 64 participating threads of each physical wave. For block 128 with all
inputs active, the two expected wave sums are 2,080 and 6,176. A block-wide
8,256 or four 32-thread sums 528 / 1,552 / 2,576 / 3,600 violates this contract.

The independent CPU oracle constructs the padded input list, takes separate
width-64 or width-32 list slices, indexes the source within each slice, and sums
each 64-element slice exactly. It uses no GPU lane-bit calculation or reduction
implementation. Every physical output is checked as an exact int32 bit pattern,
including expected zeros. All values and sums fit signed int32; no tolerance or
floating-point comparison is involved.

The device output has 32 uint32 guard words on each side, initialized together
with the payload to `0xffffffff`. This sentinel is signed -1 as payload and
differs from every expected nonnegative result, including zero. The checker
examines both guards and every payload channel, not only active input threads.
The default plan checks 19,008 payload words across 2,112 physical threads, plus
1,280 guard words. Nearby writes are covered; arbitrary invalid reads or distant
writes are not established by guards. Only final output snapshots are retained,
so repeated intermediate launches are not separately checked.

## Run procedure and resource records

Keep prepared inputs, binaries and evidence outside the source checkout. Prepare
and compile without a GPU lease. Commit the probe before execution and run its
frozen checkout through the existing allocator under an external timeout.

```sh
python3 experiments/wave_collectives/experiment.py prepare /tmp/metax-wave-input
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  bash experiments/wave_collectives/compile.sh /tmp/metax-wave-probe
mkdir /tmp/metax-wave-output
# Invoke through the existing allocator with its external process timeout:
/tmp/metax-wave-probe --run /tmp/metax-wave-input /tmp/metax-wave-output
# After exit and lease release, perform the bulk CPU oracle:
python3 experiments/wave_collectives/experiment.py check /tmp/metax-wave-input /tmp/metax-wave-output
```

Compilation explicitly uses `-x maca` and the observed `xcore1000` compiler
target, separately from the physical C550/XCORE1002 identity. An unavailable
shuffle/reduction declaration or unsupported lowering must fail compilation;
this source provides no fallback implementation. An installed header's software
helper or a successful API call does not establish one native collective
instruction. Compilation and device results are retained separately.

The executable validates the complete 512-byte input before device use and
reuses one input and one maximum output device allocation. The largest output
is `(128*9+64)*4 = 4,864` bytes; the two explicit device buffers total 5,376
bytes. Each case resets and retains all of its physical output words and guards.
The output directory must already exist and be empty. All status-returning
runtime API calls are checked; any API, file or timing failure returns nonzero.
The device process never reports CPU correctness as checked.

Raw records retain device name, PCI identity, observed wave width, properties,
and uninterpreted integer runtime/driver version results. A checked
`mcFuncGetAttributes` query before each case's warmups records function maximum
block size, register count, static shared bytes and local bytes. These remain
API observations, not native-instruction or occupancy claims. No other probe,
allocator or shared Compiler protocol is changed by this experiment.

## Timing scope

Each case uses 10 warmups followed by device synchronization, then 10 event
batches of 10 launches: 110 launches per case and 2,200 for the default plan.
Raw `event_batch_ms` and `host_enqueue_batch_us` values are retained. The default
stream's event interval can contain device idle gaps from host submission;
host enqueue includes per-launch error queries. Allocation, copies, output reset,
attribute queries and file writes lie outside the event interval.

These are descriptive timings of the complete nine-channel collective kernel,
not per-intrinsic latency, instruction throughput or an implementation comparison.
Compiler optimization may change the number of emitted instructions. There is
no GB/s interpretation. Addresses repeat; no application cache reset is applied,
and runtime cache policy is unknown. `MACA_LAUNCH_MODE`, `MACA_LAUNCH_BLOCKING`,
`MACA_DIRECT_DISPATCH`, `MACA_CACHE_PATH` and `MACA_CACHE_DISABLE` are recorded as
strings or null. Public projections must redact private paths. Profiling, if
performed, is separate; a one-case trace has 110 kernel launches and ten warmups.

This experiment does not test sparse masks, inactive source lanes, partial
physical waves, down-shuffle, xor, vote or floating-point reductions. Register
collectives do not replace shared-memory barriers or establish memory ordering.
No target-framework qualification or general C550 instruction guarantee follows
from this bounded semantic probe.

## CPU checks

Run `python3 -m unittest discover -s tests -p test_wave_collectives.py`. Tests
include explicit upper-subgroup broadcasts and complete-wave sums, logical-zero
and padded-thread outputs, wrong 32-thread/block-wide reductions, incorrect
second-wave sources, nonzero padding, unwritten physical outputs, both guards,
invalid plans, mask/boundary metadata and incomplete samples. CPU tests and
synthetic file fixtures do not execute MACA intrinsics or prove GPU correctness.
