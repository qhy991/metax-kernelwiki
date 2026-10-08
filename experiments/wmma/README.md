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

`C550_WMMA_WITNESS` defaults to zero. Mode `1` selects this witness suite;
successor modes `2` through `7` select only the separate product, sign, magnitude, scale, reciprocal and sign-transfer suites
below. All nonzero modes require paired control and prefix mode zero; the compile script
and source reject incompatible flags. The witness suite's exact TSV columns are
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

## Explicit individual-product and K-slot suite

`prepare --suite product-control` declares six isolated input patterns, all at
logical `M=N=16, K=2` with target `C[0,0]`. Only A row 0 and B column 0 can be
nonzero. Their two numerator slots are listed below; every value is divided by
16 before FP16 packing. All other input words are positive zero.

| Pattern | A numerators at K0, K1 | B numerators at K0, K1 | C00 integer numerator |
| --- | --- | --- | --- |
| `positive-k0` | `[-12, 0]` | `[-1, 0]` | `12` |
| `positive-k1` | `[0, -12]` | `[0, -1]` | `12` |
| `negative-k0` | `[1, 0]` | `[-13, 0]` | `-13` |
| `negative-k1` | `[0, 1]` | `[0, -13]` | `-13` |
| `pair-forward` | `[-12, 1]` | `[-1, -13]` | `-1` |
| `pair-reversed` | `[1, -12]` | `[-13, -1]` | `-1` |

The independent CPU oracle sums the two logical integer products and divides
by 256. It checks the target and all 255 other outputs, which must be finite
numerical zero. Either sign of output zero is accepted; input padding is still
validated bitwise as positive zero. The forward pair has the same packed words
as the earlier `isolated-c00` witness, but remains a separately declared case
in this suite. Both cases remain independently checkable under their own labels.

```sh
python3 experiments/wmma/experiment.py prepare /tmp/wmma-product-input \
  --suite product-control --order wmma-first
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  C550_WMMA_CONTROL=1 C550_WMMA_PREFIX=0 C550_WMMA_WITNESS=2 \
  bash experiments/wmma/compile.sh /tmp/wmma-product-probe
```

The compile flag is the closed enum `0|1|2|3|4|5|6|7`: zero preserves the original
non-pattern modes, one admits only `witness-control`, two admits only
`product-control`, three admits only `sign-control`, four admits only
`magnitude-control`, five admits only `scale-control`, six admits only
`reciprocal-control`, and seven admits only `sign-transfer-control` below. Mode 2 is a deliberate successor contract; it is not an
extension to the patterns accepted by mode 1. Values outside that enum fail
before compilation. Modes 1 through 7 require `C550_WMMA_CONTROL=1` and
`C550_WMMA_PREFIX=0`.

Products reuse the witness TSV column names, but `suite=product-control` and
`input_rule=isolated-ordered-products` are required. The pattern fixes the
ordered vectors, shape and target. A product binary rejects witness rows and
a witness binary rejects product rows, despite the shared header. A plan admits
one to six distinct product patterns; the seventh row is refused before its
inputs are read. The default plan lists the table order. Either kernel order
is supported; any reverse case order or profiling subset must be saved in the
TSV. A subset does not establish six-pattern coverage.

The oracle and raw protocol use experiment
`wmma-scalar-fp32-product-control`. The protocol records `witness_mode=2`, the
complete `product_patterns` table and its six-case bound. Each logical-case,
variant and snapshot record binds the suite, pattern, target and input rule,
plus `k_slots`, `a_numerators`, `b_numerators`, `product_numerators`,
`nonzero_product_k_slots` and `target_reference_numerator`. Vector metadata
is type-sensitive: Boolean or floating-point values cannot replace declared
integers. Both the packed words and this metadata must match the pattern.

The physical tile, kernel bodies, launch bodies, shared A/B allocations, separate
C allocations, guards and three input snapshots retain the paired protocol.
Each source kernel consumes one padded K16 chunk, including its fourteen zero
K slots. One full six-pattern order checks 3,072 payload words, 1,536 guards,
12,288 prepared halfwords and 36,864 snapshot halfwords; it retains 120 timed
batches and executes 1,320 launches including warmups. Per-variant timing
remains ten warmups plus ten batches of ten launches. Numerical failures remain
complete diagnostics with `performance_accepted=false` and no tolerance change.

These conditions separate each selected product, its logical K placement and
the combined dot product. Swapping the K slots changes declared input placement;
it does not reveal or prescribe the native accumulation order. The observations
do not by themselves identify an instruction sequence, rounding mechanism,
physical lane mapping or hardware-versus-compiler cause. Unchanged source
kernel bodies do not establish identical binaries or a future device outcome.

`tests/test_wmma_products.py` checks all six exact input/output contracts, mode
and suite isolation, misplaced or additional support, unannounced slot swaps,
positive-zero padding, missing and type-invalid nested metadata, complete mixed
numerical failures, snapshots and guards. Its synthetic CPU data provides no
C550 product-suite result.

## Explicit product-sign suite

`prepare --suite sign-control` admits eight fixed patterns at logical
`M=N=16, K=2`, all targeting `C[0,0]`. Only A row 0 and B column 0 contain the
listed numerator pairs, scaled by 1/16 before FP16 packing. Every other input
word is positive zero. The independent integer oracle divides the target
numerator by 256 and requires all other 255 outputs to be finite numerical zero.

| Pattern | A numerators at K0, K1 | B numerators at K0, K1 | C00 numerator | Matching earlier product pattern |
| --- | --- | --- | --- | --- |
| `positive12-k0` | `[-12, 0]` | `[-1, 0]` | `12` | `positive-k0` |
| `negative12-k0` | `[12, 0]` | `[-1, 0]` | `-12` | None |
| `positive13-k1` | `[0, -1]` | `[0, -13]` | `13` | None |
| `negative13-k1` | `[0, 1]` | `[0, -13]` | `-13` | `negative-k1` |
| `pair-pp` | `[-12, -1]` | `[-1, -13]` | `25` | None |
| `pair-pn` | `[-12, 1]` | `[-1, -13]` | `-1` | `pair-forward` |
| `pair-np` | `[12, -1]` | `[-1, -13]` | `1` | None |
| `pair-nn` | `[12, 1]` | `[-1, -13]` | `-25` | None |

The four paired cases change A's signs while keeping the complete B input
fixed. Each standalone control zeros **both** operands at the inactive K slot.
Thus the three named earlier patterns have the same complete packed inputs;
they are known baselines, not novel input conditions. Matching the declared
operands does not imply unchanged device outputs or binaries.

The factorization is part of this contract. Moving a sign from A to B while
preserving its mathematical product changes the input pattern and is refused.
These conditions study specified sign combinations and magnitudes in two
logical K slots. They do not prescribe or identify native accumulation order,
instruction selection, rounding mechanism, or a hardware-versus-compiler cause.
Standalone component observations remain separate from the combined operation.

```sh
python3 experiments/wmma/experiment.py prepare /tmp/wmma-sign-input \
  --suite sign-control --order wmma-first
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  C550_WMMA_CONTROL=1 C550_WMMA_PREFIX=0 C550_WMMA_WITNESS=3 \
  bash experiments/wmma/compile.sh /tmp/wmma-sign-probe
```

Mode 3 is an explicit successor contract and admits only `sign-control` with
`input_rule=isolated-signed-products`. Successor mode 4 selects the separate
magnitude suite below; values outside 0 through 7 are invalid. Existing modes
0, 1 and 2 retain their own suite and pattern admission. The sign TSV uses the existing
pattern columns; shared column names do not permit cross-suite rows. Sign
plans contain one to eight distinct patterns. The ninth row is refused before
reading its input files. The table order is the default; both kernel orders,
explicit case reordering and subsets are supported. A subset does not prove
coverage of all sign combinations.

The oracle and raw protocol use experiment `wmma-scalar-fp32-sign-control`.
The protocol declares `witness_mode=3`, `sign_patterns`, the fixed paired B,
the standalone inactive-slot policy and the eight-case bound. Logical-case,
variant and snapshot records retain all ordered-pair metadata and add:

- `product_signs`: integer -1, 0 or +1 at each declared K slot;
- `product_magnitudes`: absolute integer product numerators at those slots;
- `matching_product_pattern`: one of the three earlier pattern names above,
  or an explicit JSON null for the other five patterns.

The nullable field is required; omitting it is not equivalent to a declared
null. Nested vectors remain type-sensitive, so Boolean and floating-point
substitutions for integer metadata are refused. These sign-specific fields are
not added to old product or witness records. The product and sign suites share
the host-side ordered-pair parser, packing and metadata implementation, with
separate closed contract tables.

Both device kernel bodies and the launch body remain unchanged. The same full
physical tile, one padded K16 chunk, shared A/B allocations, separate guarded
C buffers, three full input snapshots and exact finite numerical oracle apply.
One complete eight-pattern order checks 4,096 payload words, 2,048 guards,
16,384 prepared halfwords and 49,152 snapshot halfwords. It retains 160 timed
batches and executes 1,760 kernel launches including warmups. No numerical
failure becomes a pass through tolerance, and timings remain descriptive with
`performance_accepted=false`.

`tests/test_wmma_signs.py` checks exact vectors and baselines, fixed
factorization, signs and magnitudes, positive-zero inputs, all physical outputs,
explicit null and deep typed metadata, guards, snapshots, mode boundaries and
complete mixed-failure diagnostics. These CPU tests use synthetic outputs and
establish no new device result.

## Explicit adjacent-magnitude suite

`prepare --suite magnitude-control` fixes logical `M=N=16, K=2` and target
`C[0,0]`, and tests every integer q from 1 through 14. Each q has three roles:

| Role | A numerators at K0, K1 | B numerators at K0, K1 | C00 numerator |
| --- | --- | --- | --- |
| `positive` | `[-q, 0]` | `[-1, 0]` | `q` |
| `negative` | `[0, 1]` | `[0, -(q+1)]` | `-(q+1)` |
| `pair` | `[-q, 1]` | `[-1, -(q+1)]` | `-1` |

Operand numerators are divided by 16 before packing; the independent integer
reference divides each C00 numerator by 256. Only A row 0 and B column 0 can
be nonzero. All other input words are positive zero, and both operands at an
inactive standalone K slot are positive zero. Every output, including all 255
non-target values, must be finite and numerically exact. Output signed zeros
remain equivalent. The largest absolute input numerator is 15 at q=14, within
the host's existing exact sixteenth-encoding domain. q=0 and q=15 are refused;
the sweep does not establish behavior outside this range.

The full plan has 42 distinct patterns: `q01-positive`, `q01-negative`,
`q01-pair`, then the same roles for q02 through q14. Its case IDs use underscores,
for example `magnitude_q01_positive`. Both kernel orders are supported; case
reordering and subsets must be explicit in the saved TSV. A subset cannot claim
coverage of all fourteen magnitude groups.

```sh
python3 experiments/wmma/experiment.py prepare /tmp/wmma-magnitude-input \
  --suite magnitude-control --order wmma-first
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  C550_WMMA_CONTROL=1 C550_WMMA_PREFIX=0 C550_WMMA_WITNESS=4 \
  bash experiments/wmma/compile.sh /tmp/wmma-magnitude-probe
```

Mode 4 admits only `magnitude-control`; successor mode 5 selects the separate scale suite below, and mode 8 is invalid. Its exact TSV columns
are `id, m, n, k, suite, pattern, q, role, target_row, target_col, input_rule, order, warmups, samples, launches`
(tab-separated). q and role must match the closed pattern name, target `(0,0)`
and `input_rule=isolated-adjacent-magnitudes`. The new header makes q and role
explicit and is rejected by old modes. Mode 4 also rejects all old headers.
A plan contains one to 42 distinct admitted patterns, and the 43rd row is
refused before its input files are read.

The oracle and raw protocol use experiment
`wmma-scalar-fp32-magnitude-control`. The protocol records `witness_mode=4`,
`magnitude_patterns`, the q range, the three roles, scales and the 42-case bound.
Each logical-case, variant and snapshot record binds integer `q` and string
`role`, ordered operand vectors, product numerators, signs, magnitudes, slots
and exact reference numerator. It also requires `matching_sign_pattern`:

- `q12-positive` maps to `positive12-k0`;
- `q12-negative` maps to `negative13-k1`;
- `q12-pair` maps to `pair-pn`;
- all other patterns declare explicit JSON null.

These three mappings state known equal packed-input contracts, not new device
comparisons. Missing nullable fields, wrong roles or magnitudes, and Boolean or
floating-point substitutions for integer metadata are refused. The factorization
also remains fixed: moving a sign between operands or permuting their K slots
under an unchanged case label fails complete packed-input validation, even if
the mathematical dot product stays the same.

The suite reuses the ordered-pair host implementation and the existing paired
executor. Both device kernels and the launch helper remain unchanged, as do
physical tile size, padded K16 execution, buffer roles, guards, three complete
input snapshots and the exact oracle policy. One full 42-pattern order checks
21,504 payload words, 10,752 guards, 86,016 prepared halfwords and 258,048 snapshot
halfwords. It retains 840 timed batches and executes 9,240 kernel launches
including warmups. Timings remain descriptive with no performance acceptance.

The paired mathematical reference stays `-1/256` while the two signed product
magnitudes change. Individual-product observations and joint observations remain
separate; no additive error model, native accumulation order, rounding mechanism,
physical lane mapping or unique hardware/compiler cause follows from this
protocol. A complete bounded sweep can describe its observed q dependence;
it cannot establish a rule for arbitrary operands, wider q ranges or GEMM.

`tests/test_wmma_magnitudes.py` covers the full table, exact packed words and
integer references, range and role refusals, explicit null and deep types,
known baselines, factorization, inactive operands, guards, snapshots, both
orders and complete numerical failures. Synthetic CPU outputs are not C550
measurement evidence.

## Explicit exact-dyadic A-scale suite

`prepare --suite scale-control` fixes `M=N=16, K=2, C[0,0]` and admits the full
45-condition grid: q in `(6,7,12)`, `scale_exp` e in `(-2,-1,0,1,2)`, and role
in `(positive,negative,pair)`. The base A pair `[-q,1]/16` is multiplied by
`2^e`; B remains `[-1,-(q+1)]/16`. Standalone roles zero both operands in their
inactive K slot. Only A row 0 and B column 0 can contain nonzero values.

| Role | Active A values | B values | Exact C00 reference |
| --- | --- | --- | --- |
| `positive` | `[-q × 2^e /16, 0]` | `[-1/16, 0]` | `q × 2^e /256` |
| `negative` | `[0, 2^e /16]` | `[0, -(q+1)/16]` | `-(q+1) × 2^e /256` |
| `pair` | `[-q × 2^e /16, 2^e /16]` | `[-1/16, -(q+1)/16]` | `-2^e /256` |

All remaining operand words are positive zero. Every output is checked, with
all 255 non-target outputs required to be finite numerical zero. Scaling is an
exact dyadic input contract: negative exponents retain fractional sixteenths,
and are never truncated to integer numerators over 16. Python preparation uses
an exact power-of-two multiplication before binary16 packing. The new native
host validator adjusts the normal binary16 exponent bits of the already exact
base value. Every admitted nonzero value remains finite and normal, including
`1/64` and `-3`. The old `sixteenth_bits` function and older packing contracts
are unchanged. This host-encoding statement does not prescribe WMMA arithmetic.

```sh
python3 experiments/wmma/experiment.py prepare /tmp/wmma-scale-input \
  --suite scale-control --order wmma-first
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  C550_WMMA_CONTROL=1 C550_WMMA_PREFIX=0 C550_WMMA_WITNESS=5 \
  bash experiments/wmma/compile.sh /tmp/wmma-scale-probe
```

Mode 5 admits only `scale-control`; successor mode 6 selects the reciprocal suite below, and mode 8 is invalid. Modes 0 through 4 retain
their existing suites. The new exact TSV columns are
`id, m, n, k, suite, pattern, q, scale_exp, role, target_row, target_col, input_rule, order, warmups, samples, launches`
(tab-separated), with `input_rule=isolated-a-power-of-two-scale`. The default
order is q ascending, then e ascending, then positive/negative/pair. Exponent
tags are `em2`, `em1`, `e0`, `ep1` and `ep2`: for example pattern
`q06-em2-positive` has case ID `scale_q06_em2_positive`. q, integer e and role
must match the closed pattern entry. A 46th row is refused before loading its
inputs. Both kernel orders, explicit case reordering and selected subsets are
supported; subsets establish no full-grid coverage.

The 45 parameter conditions contain **41 distinct complete A/B input pairs**.
Four positive-control aliases connect `(q=6,e=-1,0,1,2)` respectively to
`(q=12,e=-2,-1,0,1)`. Their complete input buffers are equal. All planned
conditions remain present and independently recorded; the aliases are not
deduplicated or counted as distinct input data.

The experiment label is `wmma-scalar-fp32-scale-control`. Every pattern,
logical-case, variant and snapshot binds q, e, role and exact rational metadata:

- `a_base_numerators` describes the unscaled A numerator pair;
- `scale_numerator=2^max(e,0)` and `scale_denominator=2^max(-e,0)` describe the A multiplier;
- `a_numerators` is the base pair multiplied by `scale_numerator`, and
  `a_denominator=16 × scale_denominator`;
- `b_numerators` is unchanged and `b_denominator=16`;
- `product_numerators` multiplies the integer A and B numerators, and
  `product_denominator=256 × scale_denominator`;
- `target_reference_numerator` is their integer sum, with
  `target_reference_denominator=product_denominator`.

Thus numerator metadata can exceed the old encoder's base range without
changing that encoder: the native scale validator separately consumes the base
numerator and exponent. Negative e uses a larger denominator, not a rounded
integer. Product signs, numerator magnitudes and occupied K slots remain
explicit. The scale oracle omits the old global input/output denominator
fields, using `base_input_denominator=16` and each pattern's rational fields.
All integer and vector metadata is type-sensitive.

`matching_magnitude_pattern` is required, including explicit null when no old
input matches. Complete declared operand words, including padding, are compared
against all 42 earlier magnitude patterns. Fourteen conditions match: all nine
e=0 conditions, plus these five positive controls:

| Scale condition | Earlier magnitude pattern |
| --- | --- |
| `q06-em1-positive` | `q03-positive` |
| `q06-ep1-positive` | `q12-positive` |
| `q07-ep1-positive` | `q14-positive` |
| `q12-em2-positive` | `q03-positive` |
| `q12-em1-positive` | `q06-positive` |

There are no further full-input matches for negative or paired roles. These
are declared input relationships; a historical device comparison must compare
the actual complete retained operands and outputs separately. Neither matching
shape nor e=0 alone is an adequate historical identity test.

Both kernel bodies and the launch helper remain unchanged. Physical tile size,
full padded K16 execution, buffer roles, 64 guards per output side, three input
snapshots and the finite exact comparison policy are retained. One complete
45-condition order checks 23,040 payload words, 11,520 guards, 92,160 prepared
halfwords and 276,480 snapshot halfwords. It retains 900 timed batches and
executes 9,900 kernel launches including warmups. Failures remain diagnostic
failures, with `performance_accepted=false` and no tolerance change.

The experiment changes A's exponent while holding B and the declared q/role
condition fixed. Any numerical scale dependence is bounded to these operands
and this execution path; it does not identify internal precision, instruction
selection, rounding or accumulation order, nor establish behavior for arbitrary
scales. Individual and paired results remain separate observations.

`tests/test_wmma_scales.py` checks all 45 complete packed inputs and references
against independent `Fraction` arithmetic and binary16 conversion, input support,
all 41 distinct input pairs, full-word historical matching, typed rational
metadata and null bindings. It also checks wrong operand scaling, negative-e
truncation, factorization changes, guards, snapshots and complete mixed numerical
failures. These CPU tests use synthetic outputs, not new C550 measurements.

## Explicit reciprocal input-scale suite

`prepare --suite reciprocal-control` keeps the physical and logical tile
`M=N=16, K=2` and target `C[0,0]`. It declares q in `(6,7,12)`, A exponent e in
`(-2,-1,0,1,2)`, B exponent `-e`, and positive/negative/pair roles. A is scaled
by `2^e` and B by `2^-e` from their base numerator pairs over 16:

| Role | Base A numerators | Base B numerators | Exact C00 reference for every e |
| --- | --- | --- | --- |
| `positive` | `[-q, 0]` | `[-1, 0]` | `q/256` |
| `negative` | `[0, 1]` | `[0, -(q+1)]` | `-(q+1)/256` |
| `pair` | `[-q, 1]` | `[-1, -(q+1)]` | `-1/256` |

Both operands at an inactive component slot are positive zero. All other
operand words are positive zero. Multiplying the exact dyadic values cancels
the reciprocal scale factors, so the two individual products and reference
sum are invariant across e. This is an input/oracle contract, not an assumed
property of the device outputs.

Mode 6 reuses the bounded `scaled_halfword` and `scaled_sixteenth_bits` encoders
for each operand, without altering either encoder or their admitted base domain.
All admitted nonzero values remain normal and finite, including maximum
`|B|=13/4`. The kernel bodies, launch helper and full padded K16 execution remain
unchanged. No new device operation or timing path is introduced.

```sh
python3 experiments/wmma/experiment.py prepare /tmp/wmma-reciprocal-input \
  --suite reciprocal-control --order wmma-first
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  C550_WMMA_CONTROL=1 C550_WMMA_PREFIX=0 C550_WMMA_WITNESS=6 \
  bash experiments/wmma/compile.sh /tmp/wmma-reciprocal-probe
```

Mode 6 admits only `reciprocal-control`; successor mode 7 selects the sign-transfer suite below, and mode 8 is invalid. Its distinct TSV
header is
`id, m, n, k, suite, pattern, q, a_scale_exp, b_scale_exp, role, target_row, target_col, input_rule, order, warmups, samples, launches`
(tab-separated). It requires `input_rule=isolated-reciprocal-power-of-two-scale`.
Both exponent fields are integers and must match the closed pattern with
`b_scale_exp=-a_scale_exp`. One-sided or same-sign exponent substitutions fail
admission. The mode-5 and mode-6 headers reject each other's inputs.

The default order is q ascending, A exponent ascending, then positive/negative/
pair. Patterns expose both exponents: `q06-am2-bp2-positive`,
`q06-am1-bp1-positive`, `q06-a0-b0-positive`, `q06-ap1-bm1-positive` and
`q06-ap2-bm2-positive` illustrate the five exponent settings. Case IDs prepend
`reciprocal_` and replace hyphens with underscores. Both kernel orders and
explicit subsets or case reordering remain available. There are 45 admitted
conditions; a duplicate pattern or a 46th row is refused before device calls.

The oracle/protocol experiment is `wmma-scalar-fp32-reciprocal-control`.
Every pattern, logical-case, variant and snapshot binds q, role, both exponents,
both base numerator vectors, and both exact scale ratios. For each operand,
`*_scale_numerator=2^max(exponent,0)` and
`*_scale_denominator=2^max(-exponent,0)`. Effective `a_numerators` and
`b_numerators` multiply their respective base vectors by those scale numerators;
`a_denominator` and `b_denominator` are 16 times their corresponding scale
denominators. Product numerators multiply those effective integer numerators,
and `product_denominator=a_denominator × b_denominator`. The target reference
uses their integer sum over that product denominator. These unreduced rational
fields state the same exact products at every reciprocal exponent. Signs,
numerator magnitudes, active slots and the target/reference remain explicit.
The new suite has no ambiguous single `scale_exp` field.

Complete-buffer comparisons establish that the 45 reciprocal conditions have
45 distinct prepared A/B pairs. Matching against all 45 prior one-sided-scale
conditions finds nine matching reciprocal conditions and eleven prior-pattern
links. The `matching_scale_patterns` field is a required list, including an
empty list where no input matches. Two lists contain both prior aliases:

- `q06-a0-b0-positive`: `q06-e0-positive`, `q12-em1-positive`;
- `q12-a0-b0-positive`: `q06-ep1-positive`, `q12-e0-positive`.

The remaining seven matches are the same-q/role e=0 patterns. These counts and
lists follow complete A/B word equality, including zero slots and padding;
no exponent-based filter is used. Actual historical projections must again
compare the complete retained input arrays and preserve every match rather
than choose one prior identity. A historical match is a declared input
relationship, not a new numerical outcome or binary-identity claim.

The exact finite-output policy, signed-zero equivalence, separate guarded C
buffers and three full A/B snapshots are retained. One full 45-condition order
checks 23,040 payload words, 11,520 guards, 92,160 prepared halfwords and 276,480
snapshot halfwords, with 900 timed batches and 9,900 launches including warmups.
All numerical failures remain failures without a tolerance change; timings are
descriptive and `performance_accepted=false`.

Reciprocal scaling changes the factorization of the same exact products. Any
observed change or invariance is bounded to these inputs and the measured
software/device path. It does not identify internal precision, rounding,
instruction selection, accumulation order or a unique hardware/compiler cause.
Standalone and paired outputs remain separate observations.

`tests/test_wmma_reciprocal.py` independently checks all 45 operands and fixed
references with `Fraction` arithmetic and binary16 conversion. It tests complete
input uniqueness and plural prior matches, both exponent bindings, wrong
one-sided and same-sign scaling, product-preserving input changes, deep typed
rational metadata, required empty/plural lists, guards, snapshots and complete
mixed numerical failures. Synthetic CPU outputs are not C550 evidence.

## Explicit product-preserving sign transfers

`prepare --suite sign-transfer-control` fixes q in `(6,7,12)`, logical
`M=N=16, K=2`, and target `C[0,0]`. The base ordered pairs are
`A=[-q,1]/16` and `B=[-1,-(q+1)]/16`. A flag `flip_k0` or `flip_k1` multiplies
**both** operands of that active term by −1, preserving each signed product.
Standalone roles zero both inactive operands and forbid an inactive flip:

| Role | Allowed `(flip_k0,flip_k1)` | Exact C00 reference |
| --- | --- | --- |
| `positive` | `(0,0)`, `(1,0)` | `q/256` |
| `negative` | `(0,0)`, `(0,1)` | `-(q+1)/256` |
| `pair` | `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` | `-1/256` |

This gives eight conditions per q and 24 total. All remaining operand words
are positive zero. The full physical output is checked: C00 must match the
exact integer reference, and the other 255 values must be finite numerical zero.
Changing only one factor's sign changes a product and is refused as a packed
input error. An undeclared two-factor sign change is also refused, even though
its mathematical product remains equal. Numerical equivalence does not replace
the declared input-pattern identity.

```sh
python3 experiments/wmma/experiment.py prepare /tmp/wmma-sign-transfer-input \
  --suite sign-transfer-control --order wmma-first
MXCC=/opt/maca/mxgpu_llvm/bin/mxcc C550_ARCH=xcore1000 \
  C550_WMMA_CONTROL=1 C550_WMMA_PREFIX=0 C550_WMMA_WITNESS=7 \
  bash experiments/wmma/compile.sh /tmp/wmma-sign-transfer-probe
```

Mode 7 admits only `sign-transfer-control`; mode 8 is invalid. Its exact TSV
columns are `id, m, n, k, suite, pattern, q, role, flip_k0, flip_k1, target_row, target_col, input_rule, order, warmups, samples, launches`
(tab-separated). It requires `input_rule=isolated-product-preserving-sign-transfer`.
Both flags must be integer 0 or 1 and match the role's closed pattern; Boolean,
floating-point, out-of-range and inactive-flag substitutions are refused.

The default order is q ascending, then the table's role and flag order. Pattern
`q06-positive-f10` flips term K0, while `q06-negative-f01` flips K1: the first f
bit is K0 and the second is K1. IDs use underscores, for example
`sign_transfer_q06_pair_f11`. Both kernel orders, explicit case reordering and
subsets remain supported. Duplicate patterns and a 25th row are refused before
device calls; a subset does not establish complete sign-transfer coverage.

The oracle/protocol experiment is `wmma-scalar-fp32-sign-transfer-control`.
Protocol metadata declares `witness_mode=7`, all 24 `sign_transfer_patterns`,
the role-specific flag pairs and their product-preserving rule. Each
logical-case, variant and snapshot binds q, role, both integer flags, the
role-masked `a_base_numerators` and `b_base_numerators`, `transfer_multipliers`,
and the actual ordered `a_numerators` and `b_numerators`. Signed products,
numerator magnitudes, occupied K slots and exact reference are explicit, with
operand denominators 16 and product/reference denominator 256. Missing fields
and deep type substitutions are refused.

Independent complete-buffer comparison establishes 24 distinct declared A/B
inputs. Nine conditions match the prior reciprocal suite: the `(0,0)` flag
condition for each q and role, each matching `qNN-a0-b0-role`. The required
`matching_reciprocal_patterns` list retains every full-input match and is empty
for the other fifteen conditions. These relationships come from all 2,048
prepared halfwords, including inactive zeros and unused chunks; actual historical
results must again compare complete retained operands and outputs. No matching
shape or unflipped-label shortcut establishes a device result.

Both kernel bodies, the launch helper and all existing encoders are unchanged.
The shared paired execution retains full padded K16 participation, separate C
buffers, 64 guards per output side and three full A/B snapshots. One complete
24-condition order checks 12,288 payload words, 6,144 guards, 49,152 prepared
halfwords and 147,456 snapshot halfwords. It retains 480 timed batches and
executes 5,280 launches including warmups. Timing remains descriptive, numerical
failures remain failures, and `performance_accepted=false` is unchanged.

The experiment varies factor signs while preserving each exact product and
reference within q/role. Any change or invariance in device outputs is bounded
to these inputs and this software/device path; it does not establish internal
precision, instruction selection, rounding, accumulation order or a unique
hardware/compiler cause. Standalone and paired results remain separate.

`tests/test_wmma_sign_transfer.py` independently checks all 24 exact inputs and
references with `Fraction` arithmetic and binary16 conversion, full-input
uniqueness/history, one-sign errors, inactive-flip admission, negative-zero
padding, typed metadata and required empty lists, guards, snapshots and complete
mixed failures. Synthetic CPU outputs are not C550 measurement evidence.
