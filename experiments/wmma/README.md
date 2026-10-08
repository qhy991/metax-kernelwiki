# Native MACA WMMA: one FP16-input, FP32-output tile

[Home](../../README.md) · [Measured diagnostics](../../wiki/wmma-exactness.md)

Question: does this installed native MACA WMMA interface compute a complete
16×16 FP32 result from FP16 A-row/B-column packed chunks, including logical
M/N/K tails and an empty reduction? This is a bounded numerical and layout
probe, not an end-to-end GEMM implementation or a throughput comparison.

## Installed interface and launch

The source explicitly includes `<mcr/mc_runtime.h>` followed by the installed
compiler resource header `<__clang_maca_mma_functions.h>`. It uses the native
`mxmaca::wmma` namespace and `__half` operand type. This is the version-bound
native SDK route. The official compatibility sample's `<mma.h>` belongs to
cu-bridge in the inspected installation; this probe uses no `nvcuda` alias,
CUDA-compatibility flag or fallback header. Missing or unsupported declarations
must fail actual MXCC compilation and remain retained evidence.

One full 64-thread block declares `matrix_a` and `matrix_b` fragments with
dimensions `16,16,16`, FP16 operands, A row-major and B column-major. Its FP32
accumulator fragment is filled with zero. Every thread follows the same
`ceil(K/16)` loop, loads one A/B chunk at leading dimension 16, and calls the
installed four-argument `mma_sync(accumulator, A, B, accumulator)`. After the
loop, every thread participates in `store_matrix_sync` to a full row-major
16×16 C tile, also with leading dimension 16. There is no early return for a
logical tail. With K=0 the loop is empty, but the filled accumulator is stored.

The executable requires exactly one visible device named `MetaX C550`, observed
runtime wave size 64 and admission of block size 64. These conditions select the
experiment; they do not establish another device's fragment behavior. Fragment
declarations or an installed compiler builtin do not alone prove native
instruction counts or performance.

## Twelve logical shapes

The prepared cases are `(M,N,K)`:

```text
(16,16,0)  (1,1,1)    (16,16,16) (15,16,16)
(16,15,16) (15,15,15) (7,9,17)   (15,16,31)
(16,15,32) (16,16,33) (9,7,63)   (16,16,64)
```

The TSV columns are `id, m, n, k, warmups, samples, launches` (tab-separated).
An explicitly saved reordered subset is admitted. Other shapes, duplicate IDs,
changed timing settings and malformed files fail before any device API call.

Each case has four fixed A chunks and four fixed B chunks, each containing
256 halfwords. Packed input files are always 1,024 uint16 words / 2,048 bytes
per operand, or 4,096 bytes per case. All host packing occurs during CPU
preparation, outside device execution and event timing.

For logical coordinates:

```text
A_num(i,k) = ((67*i + 13*k) % 31) - 15
B_num(k,j) = ((17*k + 5*j + 3) % 29) - 14
A(i,k) = A_num(i,k) / 16
B(k,j) = B_num(k,j) / 16
```

For chunk `q` and local reduction coordinate `t`:

```text
A[q*256 + i*16 + t] = A(i, q*16+t)   # row-major 16x16 chunk
B[q*256 + j*16 + t] = B(q*16+t, j)   # column-major 16x16 chunk
```

An input outside logical M/N/K, including unused chunks, is positive zero. A and
B use different coordinate formulas, so a transposed B or wrong packing is
observable. Python encodes exact half values with its standard binary16 format;
the C++ host independently verifies every stored halfword through integer half
encoding before any device API. The post-release CPU checker also validates
every packed input word against the contract.

## Independent exact oracle

The oracle uses logical `(i,j,k)` loops and the integer numerator formulas,
without multiplying decoded packed tiles or copying the WMMA indexing algorithm:

```text
C[i*16+j] = sum_k A_num(i,k)*B_num(k,j) / 256
```

Every output outside logical M/N is zero. Inputs are exactly representable in
FP16. Each product is an integer multiple of 1/256 and the absolute sum of up to
64 product numerators is bounded by `64*15*14 = 13,440`, below `2^24`. Therefore
the intended products and FP32 partial sums are exact within this bounded input
contract. This arithmetic argument does not establish arbitrary WMMA precision.

All 256 observed FP32 outputs must be finite and numerically equal to the oracle,
with positive and negative zero treated as equivalent. There is no tolerance.
The full physical tile is saved and checked even for `(1,1,1)` or K=0; logical
padding is never omitted from verification.

C has 64 uint32 guard words on **each** side. Guards and unwritten payload are
initialized to `0xffffffff`, which is a nonfinite FP32 payload and fails output
checking. Guards require exact uint32 identity. The complete output file holds
384 words / 1,536 bytes. Nearby writes are covered; arbitrary invalid reads or
distant writes are not established by guards. Only final snapshots are retained,
so intermediate repeated launches are not individually checked.

The default suite checks 24,576 input halfwords, 3,072 output floats and 1,536
guard words. One A allocation, one B allocation and one C allocation are reused
across cases: 2,048 + 2,048 + 1,536 = 5,632 explicit device-buffer bytes. H2D
copies and C reset occur outside the timed interval.

## Resource and alignment observations

Each case records tile/layout/dtype/chunk/leading-dimension metadata, block/grid
geometry and the exact input/output filenames. A checked function-attribute
query before warmups records `maxThreadsPerBlock`, `numRegs`, `sharedSizeBytes`
and `localSizeBytes`. Device identity, PCI bus ID and raw integer runtime/driver
version results are retained too.

`pointer_alignment_observed_bytes` gives the largest power of two dividing the
actual device addresses for A, B and the C payload pointer. C's prefix adds 256
bytes before its payload. These are observations about the addresses used, not
imported NVIDIA alignment rules or a claim that a particular alignment is
necessary. The checker requires a valid positive power-of-two observation but
does not impose an unverified minimum alignment.

## Procedure and timing

Keep prepared inputs, binaries and run artifacts outside the source checkout.
Preparation and explicit-target compilation need no GPU lease. Commit the
probe, use its frozen checkout, and invoke device work through the existing
allocator with an external timeout. This directory introduces no runner or
allocation policy.

```sh
python3 experiments/wmma/experiment.py prepare /tmp/metax-wmma-input
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  bash experiments/wmma/compile.sh /tmp/metax-wmma-probe
mkdir /tmp/metax-wmma-output
# Inside the existing allocator with an external process timeout:
/tmp/metax-wmma-probe --run /tmp/metax-wmma-input /tmp/metax-wmma-output
# After process exit and lease release:
python3 experiments/wmma/experiment.py check /tmp/metax-wmma-input /tmp/metax-wmma-output
```

Compilation uses `-x maca` and explicit `xcore1000` codegen, kept distinct from
the C550's physical XCORE1002 identity. The executable requires an existing
empty output directory and never overwrites an earlier run. Every status-returning
runtime call is checked; input, API, timing and retention errors return nonzero.
Bulk output validation belongs to the CPU phase after resource release.

Each case has 10 warmups and synchronization, then 10 event batches of 10
launches: 110 launches per case and 1,320 for the default suite. Raw event batch
time and host enqueue duration remain separate. Events use the default stream
and can include device idle gaps during submission. Host enqueue includes
per-launch error checks. CPU packing, allocation, transfers, reset, attribute
queries and file writes are excluded from event timing.

These are descriptive full-tile kernel timings. They are not a speedup, native
matrix-instruction throughput, or end-to-end GEMM measurement: host packing is
excluded and the logical work varies across shapes. The application performs
no explicit cache reset; runtime policy remains unknown. Raw MACA launch/cache
environment fields are retained, with private paths redacted when published.
Profiling, if collected, is separate; an isolated single-case trace expects 110
kernel launches with ten warmups. Resource attributes alone do not establish
occupancy or a performance bottleneck.

## CPU tests

Run `python3 -m unittest discover -s tests -p test_wmma.py`. Tests cover every
prepared halfword, layout direction, logical tails and extra chunks, integer
exactness, transposed B, omitted K chunks, unstored padding, every guard position,
wrong C extent, tampered metadata, nonfinite outputs and signed-zero equivalence.
These tests do not compile the SDK or execute WMMA on a device. Actual native
compilation and all-output device checking remain separate acceptance steps.

## Opt-in paired scalar control with device-input readbacks

`C550_WMMA_CONTROL=1` adds a diagnostic mode to this harness. The default macro
value is zero, and `compile.sh` accepts only `0` or `1`. The original preparation,
single-WMMA record format and strict checker remain available; its checker still
stops at the first exact failure. Earlier failed runs and their oracle are not
modified or reinterpreted by the new mode.

The control uses the same twelve logical shapes, packed FP16 input words and
integer-dot/256 reference. The source body of `wmma_tile_kernel` remains unchanged;
this does not assert that the successor compiler output or binary is identical.
A second kernel launches 256 threads, one per physical C element. It converts
the same packed A-row/B-column operands with `__half2float`, uses float operands
and a float accumulator, and computes ordinary multiplication and addition over
the full `16*ceil(K/16)` padded reduction extent. K=0 still stores every zero
output. This describes scalar source code, not forced IEEE instruction behavior,
an independent ISA implementation or a promise that the scalar control passes.
Compiler contraction and emitted instructions remain observations to inspect.

Prepare the variant order explicitly:

```sh
python3 experiments/wmma/experiment.py prepare /tmp/wmma-control-forward \
  --suite scalar-control --order wmma-first
python3 experiments/wmma/experiment.py prepare /tmp/wmma-control-reverse \
  --suite scalar-control --order scalar-first
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 C550_WMMA_CONTROL=1 \
  bash experiments/wmma/compile.sh /tmp/wmma-control-probe
```

The second command changes variant order. Reverse logical-case order separately
in the retained second TSV when that is the declared run plan. Control TSV adds
`order` after `k`; it must be `wmma-first` or `scalar-first`. The control oracle
has a distinct `wmma-scalar-fp32-input-control` experiment label and exact snapshot
and output policies. Wrong suites, unexpected orders or altered contracts are
refused rather than translated. The device binary uses the same `--run` CLI,
and CPU preparation/compilation and post-release checking stay outside the lease.

For each logical case, the device process performs this sequence:

1. Copy prepared A/B to their shared device allocations once; capture all 1,024
   halfwords of each operand as the `before` readback.
2. Reset the first variant's complete 384-word C allocation, run its ten warmups
   and ten batches of ten launches, and immediately retain all C words and guards.
3. Capture the complete A/B `between` readback, with no input rewrite.
4. Reset and execute the second variant using its separate C allocation, retain
   its complete output, and capture the complete A/B `after` readback.

A/B remain the same allocations throughout the pair. The C allocations and
files are named by variant, not execution position: `.wmma.f32` and `.scalar.f32`.
The first output is closed on disk before the second variant starts. There are
four explicit device buffers totaling 7,168 bytes. The default build still uses
the original three buffers. Pointer alignments remain recorded observations;
the control requires the device to admit a 256-thread block without importing
another vendor's alignment or occupancy rules.

Raw JSONL groups each pair under a `logical_case` record. The exact order is
`input_snapshot(before)`, first `case` plus ten `sample` records,
`input_snapshot(between)`, second `case` plus ten samples, and
`input_snapshot(after)`. Variant records carry the variant, case ID, declared
order, their own geometry, resource attributes and output file. Each snapshot
binds its case ID, declared order, phase and A/B filenames. Completion reports
logical-case and variant-execution counts separately.

The control checker validates every prepared input against the unchanged fixed
packing contract. It compares each device snapshot both with those prepared
words and with the fixed packing. Snapshot equality proves equality at the
three capture boundaries; it does not establish transient values inside a
kernel or prove that no temporary mutation occurred.

For every variant the checker retains `exact_passed`, all numerical mismatches,
finite counts and separate guard integrity. Nonfinite observations retain their
raw uint32 words and a JSON-null numerical value. Every logical case receives
both variant analyses and all snapshot comparisons, even when WMMA, scalar or
both fail. With a structurally complete run, any numeric, guard or snapshot
violation produces complete JSON on stdout with `status="diagnostic_failed"`
and exit code 1. Missing/truncated files, malformed records or contract/order
mismatches instead produce `status="error"` on stderr; they do not masquerade
as complete numerical observations. No tolerance is added.

Each logical case retains 512 C payload words, 256 guard words and 6,144 input
snapshot halfwords across three stages, plus validation of its 2,048 prepared
halfwords. Each variant still has ten warmups and 10×10 timed launches. A paired
single-case trace therefore has 220 kernel events with ten warmups in **each**
110-event kernel group; removing one cumulative prefix of twenty warmups is
incorrect. Profiling must identify the two kernel groups separately.

Timing is descriptive only. The checker always reports
`purpose="correctness_diagnostic"` and `performance_accepted=false`; it computes
no speedup. If scalar passes while WMMA fails on equal captured inputs, that
narrows the observed difference to the two execution paths. It does not by
itself assign a hardware, compiler, SDK or instruction-precision cause.

Additional tests in `tests/test_wmma_control.py` exercise both orders, all three
snapshot phases, missing/mutated snapshots, wrong suite and order, scalar indexing,
mixed passing/failing variant results, nonfinite values, guard failures, complete
failure JSON and compile-flag admission. The original strict tests remain intact.

## Explicit logical-prefix suite

`prepare --suite prefix-control` adds a separate paired WMMA/scalar diagnostic
contract. It admits exactly two logical shape families:

- `singleton`: `M=N=1`, with every integer `K` from 0 through 16;
- `dense`: `M=N=16`, with every integer `K` from 0 through 16.

The default plan therefore contains 34 logical cases. Both `--order wmma-first`
and `--order scalar-first` are supported. The default order lists singleton
K0–16, then dense K0–16; save any reversed case order explicitly in the run's
TSV. An admitted subset remains possible for bounded diagnostics or profiling.

```sh
python3 experiments/wmma/experiment.py prepare /tmp/wmma-prefix-input \
  --suite prefix-control --order wmma-first
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  C550_WMMA_CONTROL=1 C550_WMMA_PREFIX=1 \
  bash experiments/wmma/compile.sh /tmp/wmma-prefix-probe
```

`C550_WMMA_PREFIX` defaults to zero and accepts only `0` or `1`. Prefix mode
requires `C550_WMMA_CONTROL=1`; the script and source reject an incompatible
combination. Its TSV has the exact columns
`id, m, n, k, suite, family, order, warmups, samples, launches` (tab-separated).
`suite` must be `prefix-control`; the declared family must match M and N, and K
must remain within 0–16. Unknown families, nonsquare or other-sized matrices,
K=17, mixed suites and more than 34 rows are refused before device calls.

The prefix oracle and raw protocol use the distinct
`wmma-scalar-fp32-prefix-control` experiment label. Protocol metadata declares
the suite, both family names, K bounds and the deliberate 34-case limit.
Logical-case, variant and input-snapshot records each bind their suite and
family. The existing default and `scalar-control` formats still admit only
their original twelve shapes and twelve-case limit. They reject the prefix
header; prefix support does not extend their shape lists silently. Existing
helper calls retain their original behavior unless prefix mode is explicit.

Only logical data changes. The physical 16×16 tile, full-wave WMMA execution,
256-thread scalar control, input formulas and /16 scale, four packed chunks,
guards, three input readbacks and exact integer-dot/256 oracle are unchanged.
K=0 stores the zero accumulator; each K from 1 through 16 uses one source-level
WMMA step with its remaining K positions zero-padded. There is no tolerance or
performance acceptance. One complete order checks 17,408 C payload words,
8,704 guards and 208,896 snapshot halfwords, with 7,480 kernel launches.

For a fixed K, A's row 0 and B's column 0 contain the same words in both families,
so the C[0,0] reference is invariant. Other physical rows and columns are zero
in the singleton family and retain their existing formulas in the dense family.
The experiment observes whether that change affects the WMMA response; it does
not assume that singleton cases pass or that previous dense failures disappear.

After each family's complete K0–16 scan, report separately the smallest tested
K with any observed WMMA residual and the smallest tested K with a C[0,0]
residual. Retain every condition, including passing cases. These are minima
within the declared input families, not global minimal counterexamples. A
subset or incomplete scan must report its unexamined conditions rather than
claiming such a minimum. Prior failed records remain unchanged.

`tests/test_wmma_prefix.py` checks family/admission boundaries, the explicit
compile flags, both orders, complete failed-diagnostic reporting, old-parser
refusal of the new header, and the unchanged C[0,0] input/reference relationship.
These CPU checks do not establish any device outcome for the prefix suite.

## Explicit two-term witness suite

`prepare --suite witness-control` declares three input patterns at the same
logical shape `M=N=16, K=2`. The physical tile remains 16×16, the input files
still contain four packed K chunks, and each kernel executes one full padded
K16 chunk. The WMMA and scalar kernel bodies are unchanged in this successor
source; this is not a claim that separate builds produce identical binaries.

| Pattern | Target coordinate | A and B input rule |
| --- | --- | --- |
| `dense-origin` | `(13,2)` | Original numerator formulas in all logical rows and columns. |
| `isolated-origin` | `(13,2)` | Only A row 13 and B column 2 retain the ordered pairs below. |
| `isolated-c00` | `(0,0)` | Relocate those same ordered pairs to A row 0 and B column 0. |

The isolated A pair is `[-12, 1]/16` at K positions 0 and 1; the B pair is
`[-1, -13]/16` in the same order. Every other isolated input word is positive
zero, including unrelated logical rows and columns, K padding and unused
packed chunks. The isolated reference therefore has one nonzero output, at
its declared target:

`((-12)*(-1) + 1*(-13))/256 = -1/256`.

The dense reference still evaluates the original independent integer dot
product at every output coordinate. Its `(13,2)` value is also `-1/256`.
The target denotes logical matrix coordinates, not a hardware lane or fragment
mapping. Reversing both pairs would leave their mathematical dot product
unchanged, but it changes this declared input experiment and is rejected by
full packed-word validation.

```sh
python3 experiments/wmma/experiment.py prepare /tmp/wmma-witness-input \
  --suite witness-control --order wmma-first
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  C550_WMMA_CONTROL=1 C550_WMMA_PREFIX=0 C550_WMMA_WITNESS=1 \
  bash experiments/wmma/compile.sh /tmp/wmma-witness-probe
```

`C550_WMMA_WITNESS` defaults to zero and accepts only `0` or `1`. Enabling it
requires paired control and disables prefix mode; the compile script and source
reject incompatible flags. Its exact TSV columns are
`id, m, n, k, suite, pattern, target_row, target_col, input_rule, order, warmups, samples, launches`
(tab-separated). The suite must be `witness-control`. The input-rule identifiers
are `dense-formulas` and `isolated-fixed-pairs`; each pattern fixes its rule and
target coordinate. Both kernel orders are supported. A plan contains one to
three distinct patterns; an explicit subset can support a trace. A fourth row
is refused before reading its inputs. Default, old scalar-control and prefix
headers remain separate and retain their own admission rules.

The oracle and raw protocol use experiment
`wmma-scalar-fp32-witness-control`. They record the patterns, targets, input
rules, fixed isolated pairs and scale. Each logical-case, variant and snapshot
record binds `suite`, `pattern`, `target_row`, `target_col` and `input_rule` to
its case ID and declared order. Snapshot phases and filenames keep the paired
control format. Preparation, initial H2D transfer, three complete A/B readbacks,
separate guarded outputs, exact finite checks and signed-zero equivalence all
retain the existing contract. No input rewrite occurs between variants.

One complete three-pattern order checks 1,536 payload words, 768 guard words,
6,144 prepared input halfwords and 18,432 snapshot halfwords. It retains 60 timed
batches and executes 660 kernel launches including warmups. A single paired
trace still has two distinct 110-event groups, each with ten warmups. Timings
remain descriptive, and `performance_accepted` remains false regardless of
numerical outcomes.

This suite observes the response to removing unrelated rows and columns and
to relocating the same ordered pair. It does not establish a global minimum
counterexample, an internal WMMA accumulation order, a physical lane mapping,
or a hardware-versus-compiler cause. A subset establishes no complete pattern
coverage. Earlier failed records and exact oracles remain unchanged.

`tests/test_wmma_witness.py` checks closed shape and pattern admission, all
input words, wrong placement, nonzero unrelated values, reversed terms,
positive-zero input padding, every output and guard, metadata binding, both
orders and complete failed diagnostics. These CPU tests use synthetic outputs;
they establish no C550 witness result.
