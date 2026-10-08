# Single-case q7 WMMA collector

This standalone native MACA collector isolates one fixed input and retains its
complete observations. It imports no earlier probe or suite machinery. It is a
successor instrumentation contract: **one launch per variant**, zero warmups,
and no event timing, latency fields or throughput calculation.

The WMMA and scalar device kernel bodies are copied exactly from commit
`747c7b5` in `experiments/wmma/probe.cpp`. They retain one 64-thread WMMA block,
one 256-thread scalar block, a 16×16 physical output and one padded K16 chunk.
Their unchanged source bodies do not establish identical compiled binaries or
an outcome under this different launch history.

## Fixed input and independent acceptance

The little-endian input arrays each contain 1,024 binary16 words: four packed
16×16 chunks, A row-major and B column-major, with leading dimension 16.
Logical M=N=16 and K=2. Only these four words are nonzero:

| Operand | Word 0 | Word 1 | All remaining words |
| --- | --- | --- | --- |
| A | `0xb700` = `-7/16` | `0x2c00` = `1/16` | positive zero |
| B | `0xac00` = `-1/16` | `0xb800` = `-8/16` | positive zero |

The CPU checker validates every prepared and captured input word against those
fixed bits. Separately, `Fraction` arithmetic evaluates
`(-7/16)*(-1/16) + (1/16)*(-8/16) = -1/256` at C00. The other 255 physical outputs
must be numerical zero. All 256 values must be finite and exactly equal to the
reference; either sign of output zero is accepted. No tolerance or known device
failure word participates in acceptance. **Both exact outputs must pass.** An
arithmetic mismatch remains a failed observation, however small or familiar.

## Build and collect

Compilation is CPU-only and must occur outside a GPU lease. The command below
uses child-scoped empty visibility masks; the later allocator selects its own device. The script fixes the native `-x maca` route and
`xcore1000` target. If `C550_ARCH` is set, only `xcore1000` is accepted. It uses
the installed native headers `<mcr/mc_runtime.h>` and
`<__clang_maca_mma_functions.h>` with `mxmaca::wmma`; it has no cu-bridge alias,
fallback or WMMA mode flags.

```sh
MACA_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
  MXCC=/opt/maca/mxgpu_llvm/bin/mxcc MACA_PATH=/opt/maca \
  bash experiments/wmma_q7/compile.sh /tmp/q7-repro
```

The output binary must not already exist. Freeze the source before device work
and use the node's existing allocation procedure. Under that allocation, invoke
one of the following with an existing empty output directory:

```sh
/tmp/q7-repro --run wmma-first /tmp/q7-output
# Or use scalar-first in a separate fresh directory.
```

Invalid CLI syntax exits 2. An unsupported order or missing/nonempty output
directory exits 1 before any device API or output write. A valid collection
requires exactly one visible leased MACA device named `MetaX C550`, observed
wave size 64 and a device block limit of at least 256. API failures exit nonzero
and retain the partial directory; they are not successful collections.

The collector saves its fixed prepared inputs, copies them H2D once, and uses
the same A/B allocations throughout. It takes complete readbacks before the
first variant, between variants and after the second. Each variant has its own
guarded C allocation: WMMA slot 0 and scalar slot 1, independent of execution
order. Before each launch all 384 C words are set to `0xffffffff`, including
64 guards on each side and the payload sentinel. The collector launches once,
checks completion, and saves the whole output before the next variant starts.
It checks every API call and, on completion, frees all four device buffers and
synchronizes before writing the final record. No CPU oracle runs in this phase.

## Retained files and metadata

Each successful directory contains ten binary files and `raw.jsonl`:

- `prepared.a.f16`, `prepared.b.f16`: fixed input words;
- `before.a.f16`, `before.b.f16`, `between.a.f16`, `between.b.f16`,
  `after.a.f16`, `after.b.f16`: full 1,024-word device readbacks;
- `wmma.f32`, `scalar.f32`: 384 uint32 words each, including both guards.

Metadata has exactly eight ordered records: device, protocol, before snapshot,
first variant, between snapshot, second variant, after snapshot, completion.
The protocol schema is `metax-kernelwiki.wmma-q7-repro.v1`. It binds the fixed
bits, layouts, geometry, order and launch counts. Snapshot records name their
neighboring variants, including explicit null boundaries. Variant records bind
the kernel, fixed allocation slot, output file, geometry and one-launch count.
The checker refuses missing, extra, reordered or mistyped metadata.

Device name, PCI identity, visible-device count, wave size and raw runtime/driver
version integers are observed. Function resource attributes and pointer
alignments are captured before each launch. Alignment means the largest power
of two dividing the observed address; it is not an imported alignment rule.
The input alignment observations must agree across the two variants. Attributes
do not establish occupancy or instruction selection. Runtime/cache environment
values are retained as strings or null; no cache policy is inferred.

## Check after release

After the collector exits and allocation release is verified, run:

```sh
python3 experiments/wmma_q7/check.py /tmp/q7-output
```

| Exit | Result | Meaning |
| --- | --- | --- |
| 0 | `pass` on stdout | Both outputs are finite/exact and all input/guard checks pass. |
| 1 | `numeric_failed` on stdout | Complete collection and intact inputs/guards, but at least one output is unequal or nonfinite. |
| 2 | `integrity_failed` on stdout | Complete files/metadata, but prepared inputs, snapshots or guards violate the contract. |
| 2 | `error`, `error_kind=structural` on stderr | Missing/truncated files or malformed/unbound metadata prevent a complete check. |

Complete reports retain both variant analyses and every mismatch even when one
fails. Integrity and numerical statuses remain separate; corrupt input data
cannot make the reference change. A full check covers 2,048 prepared halfwords,
6,144 snapshot halfwords, 512 payload elements and 256 guards. Snapshot equality
applies only at those three capture boundaries, not transient values inside a
kernel. The scalar source declares float arithmetic but does not prescribe
emitted instructions or contraction.

`tests/test_wmma_q7.py` exercises exact passes in both orders, numerical and
integrity failures, wrong signs/layout/padding, all snapshot phases, malformed
records, role swaps, truncation, guards and signed output zeros. Its synthetic
files establish checker behavior only. This fixed-input collector does not
qualify arbitrary GEMM inputs, identify a numerical cause, or change any earlier
frozen result or oracle.
