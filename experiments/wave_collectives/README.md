# Complete wave64 integer collectives

Question: on the measured C550 installation, do direct-index shuffle and exact
integer reduction preserve the documented subgroup boundaries at lanes 31/32/63
and across two complete physical wave64 groups? The retained
[toolchain source notes](../../docs/toolchain.md) link the MACA C++ 3.5.3.x
direct-shuffle contract, including source-index wrapping within each width-sized
subgroup, and the signed integer reduction interface. These documentation and
source declarations require compilation and device evidence before becoming
local correctness observations.

The default build and preparation remain the original nine-channel experiment.
An opt-in mask-type suite, described below, uses two reduction channels and a
distinct experiment/oracle contract within the same host harness.

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

## Opt-in installed mask-type overload experiment

`C550_WAVE_MASK_TYPES=1` builds a separate two-channel kernel. Its two calls
explicitly retain the mask argument types declared by the installed SDK:

```cpp
const unsigned long mask64 = 0xffffffffffffffffUL;
const unsigned mask32 = 0xffffffffU;
output[thread * 2 + 0] = __reduce_add_sync(mask64, value);
output[thread * 2 + 1] = __reduce_add_sync(mask32, value);
```

Channel 0's expected semantic group has 64 elements; channel 1's expected group
has 32 elements under the installed half-wave compatibility overload. These are
two explicitly typed full-mask APIs, and their numeric mask values also differ.
The expected group-width contract comes from inspection of the installed
overloads; it is not a measured physical width. The observed physical runtime
wave remains 64, and all physical threads participate without an early return.
The 32-bit mask stays in an `unsigned` variable through its call. The experiment
never passes a truncated low-32-bit value typed as a 64-bit mask.

The source asserts `sizeof(unsigned)*CHAR_BIT == 32`,
`sizeof(unsigned long)*CHAR_BIT == 64`, and a 32-bit signed `int`. It also asserts
that `unsigned long` is exactly `uint64_t` and verifies both mask literal types.
The opt-in mode requires the installed `MACA_HALF_WARP_SIZE` macro to exist and
equal 32. A missing declaration, different type/width or unavailable intrinsic
causes compilation to fail; no guessed constant or alternate implementation is
substituted. Compilation alone still does not establish device semantics or
native-instruction selection.

Use matching preparation and compilation modes:

```sh
python3 experiments/wave_collectives/experiment.py prepare /tmp/metax-mask-input --suite mask-types
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 C550_WAVE_MASK_TYPES=1 \
  bash experiments/wave_collectives/compile.sh /tmp/metax-mask-probe
```

The compile flag accepts exactly `0` or `1`; omission selects `0`. All other
values are refused before the compiler runs, and the selected value is passed
explicitly with `-D`. Default `prepare` and a mode-0 binary retain the original
20 cases, nine-channel kernel, record fields and oracle. The mode-1 preparation
has the same 20 block/logical-length cases under distinct `mask_types_b...` IDs,
and `wave64-int32-mask-types` identifies its oracle and raw protocol. The checker
selects a mode only from an exact recognized oracle contract; unknown or mixed
contracts fail instead of being translated.

New outputs are thread-major, `output[thread * 2 + channel]`. Their protocol and
case records contain a `channel_contracts` list with each channel's `mask_type`,
`mask_hex` string, `mask_bits` and declared `group_width`. They do not carry a
misleading single-mask field. `wave_size=64` remains separate. The opt-in raw
protocol records `mask_types_mode=1`. Old records require none of these new
fields and remain readable with the existing default helper calls.

The CPU oracle independently sums 64-element and 32-element list slices of the
same padded input. Full block 128 expects 2,080 / 6,176 in channel 0 and
528 / 1,552 / 2,576 / 3,600 in channel 1. Zero logical lengths and all physical
tail outputs remain checked. Input, guards, runtime checks and the
10-warmup + 10×10 timing contract are unchanged. The 20-case mask-type suite
checks 4,224 payload words plus 1,280 guards. Its largest output allocation is
1,280 bytes, alongside the same 512-byte input.

Because the channels request different group operations, neither their times
nor comparison with the nine-channel kernel is a speedup for one operator.
Timing remains descriptive of the complete two-channel kernel. Full masks with
two fixed types do not establish arbitrary-mask participation, inactive-source
behavior or memory ordering. Device observations must be bound to the installed
header/toolchain and the successor source commit.

Additional CPU tests check exact two-channel sums, 95/96/97 tails, swapped
channels, incorrect group boundaries, lost upper-half results, nonzero padding,
unwritten outputs, strict per-channel metadata and compile-flag admission. The
existing nine-channel tests remain as regressions; synthetic fixtures are not
device evidence.
