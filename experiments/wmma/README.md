# Native MACA WMMA: one FP16-input, FP32-output tile

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
