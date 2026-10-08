# Exact FP32 transpose: direct and shared-memory tiles

Question: on the measured C550 software stack, how do direct indexing and two
shared-memory tiles compare for the **same** row-major transpose semantics?
For input `A` with `rows` rows and `cols` columns, every variant must produce
`B[col * rows + row] = A[row * cols + col]`. These variants can be compared as
implementations of one operation. Results do not establish a general transpose
optimum or replace framework-level evaluation.

## Cases and variants

The default plan is shape-major, with three variants for each of 19 shapes:

- `(1,1), (1,65), (65,1), (31,33), (33,31)`;
- every pair in `{63,64,65} × {63,64,65}`;
- `(262144,64), (65536,256), (4096,4096), (262143,63), (4095,4097)`.

All products are at most `2^24`. The TSV columns are
`id, rows, cols, variant, warmups, samples, launches` (tab-separated). Host
admission accepts 1–57 rows from this fixed shape/variant set, allowing an
explicitly saved reordered subset or a single case for a trace. Invalid shapes,
variants, duplicate IDs or timing settings fail before any device API call.

| Variant | Block `(x,y,z)` | Source storage | Intended static shared bytes |
|---|---|---|---:|
| `direct` | `(256,1,1)` | One output per thread, runtime division and remainder to locate input | 0 |
| `tile64` | `(64,4,1)` | `float tile[64][64]` | 16,384 |
| `tile64_pad1` | `(64,4,1)` | `float tile[64][65]` | 16,640 |

The direct kernel visits output linear index `i` and reads
`A[(i % rows) * cols + i / rows]`. Tiled variants load 64×64 input regions:
each thread handles offsets `0,4,...,60` along the row direction. Every thread
then reaches one unconditional `__syncthreads`, including threads outside a
matrix edge. The store phase exchanges tile coordinates, with its own output
row and column guards. A valid store from `tile[x][y+offset]` has exactly the
same matrix-validity condition as the earlier load that initializes that slot.
No invalid shared slot is used to fill an edge.

The extra shared column is an experimental change. It is not assumed to remove
bank conflicts or improve occupancy. Before each case's warmups and timed
launches, checked `mcFuncGetAttributes` records `maxThreadsPerBlock`, `numRegs`,
`sharedSizeBytes` and `localSizeBytes` under
`function_attributes_before_timing`. Each case separately records intended
static shared bytes, tile size, padding and grid/block x/y/z. The checker accepts
a nonnegative actual shared size that differs from the intended source size;
compiler optimization or allocation behavior is a question for the result.
Function and device attributes are API observations, not occupancy measurements.

## Shared-pitch control suite

`prepare DIRECTORY --suite shared-pitch` writes a separate 38-case plan: the
same 19 shapes, each with `runtime_pitch64` and `runtime_pitch65`. The default
`prepare DIRECTORY` remains the original 57-case, three-variant plan. The TSV,
transpose oracle, buffers, guards and 10-warmup + 10×10 timing protocol remain
unchanged. The control plan therefore launches 4,180 kernels in total.

Both new variants call the **same non-template**
`transpose_runtime_pitch_kernel(rows, cols, pitch)` with runtime pitch 64 or 65.
Its source declares one fixed `float storage[64*65]`, or 4,160 elements / 16,640
bytes, for both pitches. It loads `storage[(y+offset)*pitch+x]`, reaches the same
unconditional full-block barrier, and stores from `storage[x*pitch+y+offset]`.
The 64×4 block, grid mapping, coordinate guards and transpose semantics match
the tiled variants. Host admission maps only the two named variants to 64 and
65; there is no arbitrary pitch input or pitch-specific compiled kernel.

Each new variant records `shared_pitch_elements` (64 or 65),
`allocated_shared_elements` (4,160) and `intended_static_shared_bytes` (16,640).
Its `padding` field describes logical row spacing, zero or one. Pitch 64 does
**not** mean a 16,384-byte source allocation in this suite. Older variants and
their records need no new fields and remain readable.

This control asks how changing shared row pitch behaves with one source kernel
and one declared storage capacity. Equal source capacity does not establish
equal actual shared allocation, register use, emitted instructions, residency
or hardware behavior. Preserve the per-case runtime attributes even when they
differ from the intended size or between the two pitches; later compilation,
device and trace observations must establish those properties. Neither a pitch
change nor a timing difference alone proves bank conflicts or their removal.

## Dynamic-shared control suite

`prepare DIRECTORY --suite dynamic-shared` writes 57 cases: each of the same
19 shapes with the following three explicitly mapped variants. All use one
non-template `transpose_dynamic_shared_kernel` with `extern __shared__ float
storage[]`, the same guarded indexing and unconditional barrier, and the same
64×4 block and grid. The runtime pitch and third kernel-launch argument are the
only controls changed within this suite.

| Variant | Runtime pitch, elements | Requested dynamic shared bytes |
|---|---:|---:|
| `dynamic_pitch64_bytes16384` | 64 | 16,384 |
| `dynamic_pitch64_bytes16640` | 64 | 16,640 |
| `dynamic_pitch65_bytes16640` | 65 | 16,640 |

The first pair holds the access pattern and pitch fixed while changing the
launch's requested shared-memory reservation. The second and third variants
hold requested capacity fixed while changing pitch. These comparisons preserve
the same output semantics, input, oracle, buffers, guards, warmups and timing
contract; the complete dynamic suite has 6,270 launches. The original default
57-case and static shared-pitch 38-case preparations and records are unchanged.

The host admits only these named pairs; there is no arbitrary pitch or byte
argument in the TSV or CLI. It checks that a mapped request can contain 64 full
rows at its pitch before the first device API. An insufficient pair, including
pitch 65 with only 16,384 bytes, has no admitted route. This is an experiment
capacity check, not a model of the hardware's reservation granularity or limits.

Dynamic case records separate four quantities:

- `shared_pitch_elements` is the runtime addressing pitch, 64 or 65.
- `requested_dynamic_shared_bytes` is the third launch argument; its element
  count is `requested_shared_elements`. Neither reports actual hardware allocation.
- `intended_static_shared_bytes=0` describes this kernel's source declaration.
- `function_attributes_before_timing.sharedSizeBytes`, `numRegs` and
  `maxDynamicSharedSizeBytes` retain the raw function API observations alongside
  the existing fields. `sharedSizeBytes` is the function's reported static shared
  size; `maxDynamicSharedSizeBytes` is its reported dynamic shared limit, not
  the bytes requested or allocated for a particular launch.

No attribute is forced to equal a source intention or the launch request. The
probe does not call `mcFuncSetAttribute` to change a limit. Device execution and
trace evidence must establish the actual route and behavior. Equal requested
bytes do not prove equal hardware allocation, occupancy or performance, and a
timing difference does not by itself identify bank conflicts.

## Independent CPU oracle and memory

Input `i` is `float32(i)` for `0 <= i < 16,777,216`, giving unique, finite,
exactly representable values. The CPU oracle iterates nested input `(row,col)`
coordinates and checks the independent row-major relationship
`output[col * rows + row] == input[row * cols + col]`. It uses neither the GPU
direct kernel's output-index division/remainder nor its tile algorithm. A 2×3
input `[0,1,2,3,4,5]` must yield `[0,3,1,4,2,5]`, which distinguishes the
transpose direction on nonsquare inputs.

The checker scans every input value, every final output bit pattern, finite
payload counts and 32 guard words on each side. Output payload and guards start
as `uint32 0xffffffff`; unwritten payload therefore fails. Exact copy semantics
require no floating-point tolerance. Guard checks cover nearby writes, not
arbitrary invalid reads or distant writes. Final snapshots do not establish the
correctness of every intermediate repeated launch.

One 64 MiB input device allocation and one 64 MiB + 256 byte output allocation
remain allocated for the whole process. Their base addresses and the output
payload pointer remain fixed across cases. Peak explicit device buffers are
128 MiB + 256 bytes. Before each case, only its `rows*cols+64` output words are
reset, outside timing. No case changes its input contents. Large square,
nonsquare and odd-sized matrices share the same numerical contract; boundary
cases include dimensions below, equal to and above one tile.

## Run procedure

Keep generated inputs, binaries and results outside the source checkout.
Commit the probe and execute its frozen checkout. Preparation, explicit-target
compilation and CPU checking need no GPU lease. Reuse the node's existing
allocator and the installed gpu-infra lifecycle; this experiment adds no runner
or allocation mechanism.

```sh
python3 experiments/transpose/experiment.py prepare /tmp/metax-transpose-input
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  bash experiments/transpose/compile.sh /tmp/metax-transpose-probe
mkdir /tmp/metax-transpose-output
# Invoke through the existing allocator with its external process timeout:
/tmp/metax-transpose-probe --run /tmp/metax-transpose-input /tmp/metax-transpose-output
# After the process exits, release the lease and check on the CPU:
python3 experiments/transpose/experiment.py check /tmp/metax-transpose-input /tmp/metax-transpose-output
```

Compilation uses `-x maca` and an explicit observed architecture; the inspected
`xcore1000` compiler target remains distinct from the device's XCORE1002 identity.
The executable requires exactly one visible device named `MetaX C550` and an
existing empty output directory. It records PCI bus ID, runtime device
properties, and raw integer `mcRuntimeGetVersion`/`mcDriverGetVersion` results
without interpreting their encoding. API failures, file errors, malformed
timing and incomplete runs return nonzero. CPU oracle acceptance is separate
from process exit status. A retry uses a fresh output directory.

## Timing and interpretation

Each case has 10 warmups, one device synchronization, and 10 event batches of
10 launches: 110 launches per case, 6,270 for the default plan. Raw JSONL keeps
all `event_batch_ms` and `host_enqueue_batch_us` samples. Default-stream events
bracket each batch; their interval can include device idle gaps from host
submission. Host enqueue includes the per-launch error query. Batch averages
are not isolated kernel latency or pure launch overhead. Allocation, output
reset, attribute queries, transfers and file writes are outside event timing.

Addresses repeat and the application performs no explicit cache reset. Effective
runtime cache policy is unknown. The raw protocol records `MACA_LAUNCH_MODE`,
`MACA_LAUNCH_BLOCKING`, `MACA_DIRECT_DISPATCH`, `MACA_CACHE_PATH` and
`MACA_CACHE_DISABLE` as strings or null when unset. Public projections must
redact private paths. Those environment values do not prove cache behavior.
Logical traffic is `8*rows*cols` bytes per launch; logical GB/s is neither
measured DRAM traffic nor hardware bandwidth. Padding, shared storage and runtime
register counts alone cannot establish bank conflicts, occupancy or a bottleneck.

Save the exact case order and balance variant order in independent processes
when comparing timings. The default shape-major order alone does not control
thermal or temporal effects. Keep profiling separate from unprofiled timing.
An isolated single-case trace has `--expected-launches 110 --warmups 10`; do not apply one
cumulative warmup removal to all 57 cases. Profiler startup without verified GPU
events is not a successful trace. Record frequency and concurrency limitations
with each run rather than inferring them from this source.

## CPU validation

Run `python3 -m unittest discover -s tests -p test_transpose.py`. Tests cover
nonsquare transpose direction, full output checks on small tile-boundary shapes,
wrong ordering, unwritten/nonfinite payload, guards, plan admission and missing
or invalid metadata/samples. A CPU model enumerates every local tile edge class
in the admitted shapes, verifies unique loads and stores, and verifies that every
valid shared-memory read has a prior load after the modeled full-block barrier.
Multi-tile edge cases additionally cover the complete global index sets. This
checks the algorithm's index and barrier contract; it does not execute compiled
GPU synchronization or prove MXCC code generation.

For the shared-pitch controls, the CPU model uses the same 4,160-element storage
capacity for both runtime pitches and checks every admitted local tile edge
class for in-range unique loads, initialized shared reads, unique stores and
the independent transpose result. Protocol tests reject absent, mistyped or
incorrect pitch/capacity metadata, while retaining actual API attributes as
observations. These checks add no GPU or occupancy evidence.

Dynamic-control tests check all three admitted pitch/capacity pairs against the
same local edge classes, reject insufficient or unlisted pairs before device
use, reject missing or tampered requested-capacity metadata, and require the
new dynamic-limit API field only for dynamic cases. They continue to read old
default and static shared-pitch records without backfilling new resource fields.

Python reuses only existing native CPU input validation, binary reading, endian
checks and input/guard constants. Transpose semantics and its experiment protocol
have separate ownership here; existing probes remain unchanged.
