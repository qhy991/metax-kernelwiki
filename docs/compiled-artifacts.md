# Inspecting compiled MACA artifacts

[Home](../README.md) · [Toolchain](toolchain.md) · [q7 reproducer](../experiments/wmma_q7/README.md)

The retained q7 executable contains **both LLVM bitcode and a native device ELF**. Installed SDK tools extract both payloads and decode the bitcode into LLVM IR. This identifies what the executed host file packages; runtime payload selection and final native arithmetic remain unresolved.

This page records a CPU-only inspection of the executable built from source [`9b7bef6`](https://github.com/qhy991/metax-kernelwiki/commit/9b7bef6ea79e7d71394f92a32987779d8fc18274) for the [single-launch q7 study](../wiki/wmma-exactness.md#standalone-q7-reproducer-one-launch-per-variant). It used MACA 3.5.3.18, MXCC `1.0.0 (6477545d4d)` and `-offload-arch=xcore1000`. The executable was neither rebuilt nor run during inspection. Its [bounded inspection record](../data/inspections/20261008-q7-binary.json) is separate from the unchanged [numerical result](../data/results/20261008-wmma-q7-repro.json).

## What is in this executable?

The 88,880-byte host ELF has a `.mc_fatbin` section at file offset 36,864, spanning 34,629 bytes. Its bundle header describes three entries. Offsets below are relative to that section:

| Bundle target ID | Offset | Bytes | Observed contents |
| --- | ---: | ---: | --- |
| `host-x86_64-unknown-linux-gnu` | 4,096 | 0 | Empty entry |
| `maca-mxc-metax-macahca--xcore1000-bc` | 4,096 | 11,216 | Wrapped LLVM bitcode |
| `maca-mxc-metax-macahca--xcore1000` | 16,384 | 18,232 | Device ELF |

The descriptor table ends at byte 202. Both nonempty ranges fit within the section and do not overlap. The bitcode wrapper declares a 20-byte offset to 11,184 bytes of bitcode, followed by 12 zero padding bytes. These observations agree with the upstream [binary bundle layout][bundle] and [bitcode wrapper format][bitcode]. The installed MetaX tools establish support for these particular target IDs; generic upstream LLVM documentation alone does not.

`xcore1000` here is the compiled code family. The tested C550's reported physical `xcore1002` identity is a separate fact; see [device identity](../wiki/device-identity.md).

## Extract without changing the retained input

These commands show the successfully used SDK routes, with fresh output paths. Set `RETAINED_REPRO` to the already executed file and `INSPECTION_DIR` to a directory that does not yet exist. The explicit host-copy output matters: `llvm-objcopy` modifies its input in place when the normal output operand is omitted. See its [command reference][objcopy].

```sh
set -eu
: "${RETAINED_REPRO:?Set the absolute path of the retained executable}"
: "${INSPECTION_DIR:?Set a fresh absolute output directory}"
test -f "$RETAINED_REPRO"
mkdir "$INSPECTION_DIR"
SDK_BIN=/opt/maca-3.5.3/mxgpu_llvm/bin
export MACA_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=

"$SDK_BIN/llvm-objcopy" \
  --dump-section=".mc_fatbin=$INSPECTION_DIR/fatbin.bundle" \
  "$RETAINED_REPRO" "$INSPECTION_DIR/host-copy.elf"
"$SDK_BIN/clang-offload-bundler" --type=bc --list \
  --input="$INSPECTION_DIR/fatbin.bundle"
"$SDK_BIN/clang-offload-bundler" --type=bc --unbundle \
  --input="$INSPECTION_DIR/fatbin.bundle" \
  --targets=maca-mxc-metax-macahca--xcore1000-bc \
  --output="$INSPECTION_DIR/embedded.bc"
"$SDK_BIN/clang-offload-bundler" --type=bc --unbundle \
  --input="$INSPECTION_DIR/fatbin.bundle" \
  --targets=maca-mxc-metax-macahca--xcore1000 \
  --output="$INSPECTION_DIR/embedded-device.elf"
"$SDK_BIN/llvm-dis" "$INSPECTION_DIR/embedded.bc" \
  -o "$INSPECTION_DIR/embedded.ll"
readelf -h -SW -n "$INSPECTION_DIR/embedded-device.elf"
"$SDK_BIN/llvm-nm" --format=posix "$INSPECTION_DIR/embedded-device.elf"
```

`--type=bc` is the verified binary-bundle handler invocation in this installed tool. The selected native payload is still an ELF file. `llvm-dis` produces [textual LLVM IR][dis], not a native GPU instruction listing.

All extraction and decoding commands exited zero. The original executable remained unchanged, and both unbundled files matched their declared byte ranges. Independent bounded parsing agreed with the extracted sections, payloads, symbols and metadata. Full binaries, IR and command logs remain in private retained evidence; the public record contains selected facts and commands with run paths normalized. It does not distribute the retained executable.

## What the embedded IR establishes

The decoded module has target triple `mxc-metax-macahca`. Its kernel functions use `metaxgpu_kernel`, with target CPU and feature `xcore1000`.

The WMMA function calls `llvm.mxc.mma.f32.16x16x16f16` with two `<4 x half>` inputs and a `<4 x float>` accumulator/result. There are five static call sites: one in a remainder loop and four in an unrolled loop. This count is a property of the IR control flow, not a count of operations executed by the q7 launch or native machine instructions. The intrinsic's spelling and types do not specify internal rounding or precision.

The scalar function converts halfwords through `llvm.convert.from.fp16.f32` and contains eight static pairs of `fmul contract float` and `fadd contract float`. LLVM's [`contract` flag][contract] permits contraction such as fused multiply-add. It therefore does not prove that separate multiply/add instructions or an FMA executed. This observation is consistent with the scalar control's original source-level scope.

These IR statements come from bitcode extracted from the retained executable. A new `-emit-llvm` compilation would be a different observation.

## Native symbols and metadata

The native payload has raw ELF `e_machine=253`. A generic machine-table label for this number does not qualify a MetaX instruction decoder. No forced architecture override was used.

The `MetaX` note has raw type 48 and a 3,493-byte MessagePack descriptor containing `macahca.version=[1,0]`. Selected fields and symbol extents are:

| Field | WMMA | Scalar |
| --- | ---: | ---: |
| Kernel symbol bytes | 1,304 | 1,352 |
| `.mtreg_count` | 28 | 36 |
| `.streg_count` | 20 | 19 |
| `.max_block_size` | 512 | 512 |
| `.private_memory_size` | 0 | 0 |
| `.share_memory_size` | 0 | 0 |
| `.kernarg_size_bytes` | 112 | 112 |

Symbol extents are byte sizes, not instruction counts. The `.mtreg_count` values and block attribute agree with the relevant retained runtime/profile fields; `.streg_count` is reported here only from the note. They do not establish hardware limits, measured occupancy or runtime payload selection. In particular, the [512-thread attribute](../wiki/launch-bounds.md) must remain distinct from the device's launch limit. This inspection does not qualify the full argument ABI from the note.

## Remaining coverage

System GNU `objdump` 2.38 failed automatic disassembly with exit 1 and `can't disassemble for architecture UNKNOWN!`. The checked SDK directory and `PATH` had no `llvm-objdump`, `llvm-mc`, `mxc-objdump`, `mc-objdump` or `mx-objdump`; `mxcc --help-hidden` was rejected as an unknown argument. These are limits of the tested routes, not proof that no suitable decoder exists.

The next useful evidence would identify runtime payload selection and any post-load transformations, or qualify a native decoder for this C550/SDK pair. The retained trace's false recompilation flags alone do not select between the packaged bitcode and native ELF. The q7 exactness failure remains unexplained and performance remains unaccepted. This CPU-only inspection adds no device execution, numerical acceptance or performance measurement.

[bundle]: https://releases.llvm.org/19.1.0/tools/clang/docs/ClangOffloadBundler.html#bundled-binary-file-layout
[bitcode]: https://releases.llvm.org/19.1.0/docs/BitCodeFormat.html#bitcode-wrapper-format
[objcopy]: https://releases.llvm.org/19.1.0/docs/CommandGuide/llvm-objcopy.html
[dis]: https://releases.llvm.org/19.1.0/docs/CommandGuide/llvm-dis.html
[contract]: https://releases.llvm.org/19.1.0/docs/LangRef.html#fast-math-flags
