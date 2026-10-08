# q7 through explicit native and bitcode images

[Home](../../README.md) · [Artifact inspection](../../docs/compiled-artifacts.md) · [Original q7 collector](../wmma_q7/README.md)

This host-only collector tests two explicitly selected module images from the retained q7 executable. It contains no device kernel source. The question is whether a carrier containing only the retained bitcode can load and reproduce the result, alongside a predeclared native-ELF control. It never falls back to another image kind after a failure.

This is protocol **v2**. Earlier native-module results use source `b463205`, protocol v1 and `loaded-image.elf`; replay those records with the [frozen v1 guide](https://github.com/qhy991/metax-kernelwiki/blob/b463205eff646cd0c88fe26fd51838be1f8f1726/experiments/wmma_module/README.md) and checker. The current checker does not translate old records into v2.

## Two bounded image forms

| Required image kind | Examined form | Input boundary |
| --- | --- | --- |
| `native-elf` | 18,232-byte ELF64 little-endian, ET_DYN, raw machine 253 | The extracted native payload from q7 source `9b7bef6` |
| `retained-bitcode-bundle` | 15,324-byte Clang offload bundle made by the installed MetaX tool | One empty host entry plus the exact 11,216-byte wrapped bitcode from that same executable; no native entry |

The bitcode carrier has two descriptors: `host-x86_64-unknown-linux-gnu` at offset 4,096 with size zero, and `maca-mxc-metax-macahca--xcore1000-bc` at offset 4,096 with size 11,216. Its descriptor table ends at byte 145, followed by zero padding. The payload has a 20-byte wrapper, 11,184 bytes of bitcode and 12 zero padding bytes. The carrier ends with the 12-byte `__FILE_END__` marker **without NUL**. This differs from the old embedded fatbin's 13-byte tail and is the actual standalone bundler output.

These are closed checks for two retained forms, not a general ELF or bitcode verifier. A raw bitcode file, mixed native/bitcode bundle, extra entry, changed layout or another SDK output is outside this protocol. `xcore1000` remains the compiled family; it does not rename the physical C550 target.

The collector retains the complete supplied buffer as `loaded-image.bin` and passes that buffer to `mcModuleLoadData`. It resolves the original WMMA/scalar symbols and uses the same collection and launch code for both kinds. Four argument addresses represent A, B, output payload and unsigned chunk count 1; `extra` is null. The runtime owns packing and hidden arguments. The [3.5.3 API warning](https://developer.metax-tech.com/api/client/document/preview/990/split_files/mxmaca_运行时api模块.html) conflicts with installed and [published mcTriton examples](https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.py); v1 already measured this argument-array convention for the native image. Each new route still requires its own load, lookup, launch and correctness evidence.

## Prepare the carrier without a GPU lease

Use the wrapped bitcode extracted from the retained executed q7 host file, as described in the [artifact guide](../../docs/compiled-artifacts.md). Do not recompile the source or assemble newly emitted LLVM IR as a substitute. The installed bundler reported version `1.0.0 (6477545d4d)` and accepted this command:

```sh
set -eu
: "${RETAINED_BITCODE:?Set the independently retained wrapped-bitcode path}"
: "${PACKAGE_DIR:?Set a fresh absolute directory}"
mkdir "$PACKAGE_DIR"
: > "$PACKAGE_DIR/empty-host"
MACA_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
/opt/maca-3.5.3/mxgpu_llvm/bin/clang-offload-bundler \
  --type=bc --bundle-align=4096 \
  --targets=host-x86_64-unknown-linux-gnu,maca-mxc-metax-macahca--xcore1000-bc \
  --input="$PACKAGE_DIR/empty-host" --input="$RETAINED_BITCODE" \
  --output="$PACKAGE_DIR/retained-bitcode.mcfb"
```

Bundling is the tool's default operation. This packages existing bytes and performs no device-source or LLVM-IR compilation. The empty host descriptor follows the retained bundle convention; this experiment does not establish that every loader requires it.

Validate the actual carrier before GPU admission:

```sh
python3 experiments/wmma_module/check.py \
  --inspect-image "$PACKAGE_DIR/retained-bitcode.mcfb" \
  --image-kind retained-bitcode-bundle --expected-bitcode "$RETAINED_BITCODE"
```

This CPU command checks the descriptor layout, bounds, padding, wrapper and trailer, then compares the entire bitcode payload with the independent retained reference. A structural/payload pass establishes no loader or numerical result. Keep the build command, diagnostics, output and inspection receipt under a fresh evidence directory.

Compile the host driver without a GPU lease:

```sh
MACA_PATH=/opt/maca-3.5.3 \
MACA_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
bash experiments/wmma_module/compile.sh /absolute/fresh/build/module-repro
```

Retain the frozen source commit, host compiler version and complete build command. CPU masks belong only to these children; the allocator owns the later device mapping. Read the [methodology](../../docs/methodology.md) and [GPU lease lifecycle](https://github.com/qhy991/gpu-infra/blob/main/skills/gpu-infra/SKILL.md#gpu-lease-lifecycle). Module loading may involve runtime compilation internally; keep that device-context-dependent operation in the bounded worker instead of claiming it was completed during packaging.

## Collect under the existing allocator

The planned native control is a separate declared run, never a fallback. Use a fresh empty output directory for each collection. Inside an admitted worker:

```sh
/absolute/frozen/build/module-repro --run wmma-first retained-bitcode-bundle \
  /absolute/retained/retained-bitcode.mcfb /absolute/fresh/bitcode-output
```

For the separately declared native control, select `native-elf` with its retained ELF and another fresh output directory. Use `scalar-first` for the opposite bitcode order, under a new admitted worker. Any API or integrity failure stops later stages; it does not select raw bitcode or another carrier automatically.

Both kinds preserve the [fixed q7 numerical contract](../wmma_q7/README.md): 1,024 halfwords per operand, one padded K16 step, 256 FP32 outputs, 64 guards per side and complete input snapshots before/between/after implementations. WMMA uses one 64-thread block and scalar one 256-thread block. The same A/B allocations are uploaded once, and outputs are separately initialized and retained. There are no warmups or collector event timers. The exact oracle remains -1/256 at C00 and zero elsewhere; the observed residual is never an expected value.

Protocol v2 binds the image kind in module, protocol and completion records. Its environment fields include the actual value or unset state of `MACA_MODULE_LOADING`. An error names the failing API expression. Successful collection unloads the module and frees buffers; verify process and allocator release before CPU checking. A trace is a separate diagnostic run, with exporter units unverified.

## Check after release

For the bitcode collection, require both independent image references:

```sh
python3 experiments/wmma_module/check.py /absolute/retained/bitcode-output \
  --image-kind retained-bitcode-bundle \
  --expected-image /absolute/independently-retained/retained-bitcode.mcfb \
  --expected-bitcode /absolute/independently-retained/embedded.bc
```

For native mode, use `--image-kind native-elf` and its independent ELF as `--expected-image`; omit `--expected-bitcode`. The checker compares all supplied image bytes, validates the selected form and, for the bundle, compares its complete payload against the independent original wrapped bitcode. It rejects self/hard-link reference comparisons and inconsistent route metadata. Numerical/file checks share the original q7 oracle, while each module protocol is checked separately.

Exit 0 means complete exact success; exit 1 retains a complete numerical failure; exit 2 denotes an integrity or structural failure. Actual image-input identity does not prove unchanged final instructions, a cold cache, a specific compilation mechanism, the old fatbin's selected entry or the arithmetic cause. Preserve failures and partial metadata as their own outcomes. A successful load is separate from numerical acceptance and performance acceptance.
