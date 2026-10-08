# Inspecting compiled MACA artifacts

[Home](../README.md) · [Toolchain](toolchain.md) · [q7 reproducer](../experiments/wmma_q7/README.md)

The retained q7 executable contains **both LLVM bitcode and a native device ELF**. Installed SDK tools extract both payloads and decode the bitcode into LLVM IR. Separate module probes now load each extracted payload directly. This identifies the supplied input for those probes; the earlier host executable's payload selection and final native arithmetic remain unresolved.

The original inspection examines the executable built from source [`9b7bef6`](https://github.com/qhy991/metax-kernelwiki/commit/9b7bef6ea79e7d71394f92a32987779d8fc18274) for the [single-launch q7 study](../wiki/wmma-exactness.md#standalone-q7-reproducer-one-launch-per-variant). It used MACA 3.5.3.18, MXCC `1.0.0 (6477545d4d)` and `-offload-arch=xcore1000`. The executable was neither rebuilt nor run during inspection. Its [bounded inspection record](../data/inspections/20261008-q7-binary.json) is separate from the unchanged [numerical result](../data/results/20261008-wmma-q7-repro.json). Later sections cover [producer output](#mcrtc-returns-a-wrapped-bitcode-buffer-in-a-separate-cpu-probe), [module-input routes](#separately-measured-module-input-routes) and [retained runtime caches](#retained-runtime-caches-contain-distinct-native-artifacts).

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

A later, separate [native-module experiment](../wiki/wmma-exactness.md#successor-explicit-native-elf-module-loading) supplies only the extracted native ELF to a host-only driver. It successfully loads and launches both kernels and retains the same q7 residual. That device evidence belongs to its own frozen source and result; it does not establish the earlier fatbin's payload choice or final instruction identity.

The subsequent [bitcode-carrier experiment](../wiki/wmma-exactness.md#successor-bitcode-only-carrier-rejected-before-execution) packages only the unchanged wrapped bitcode plus an empty host descriptor. The installed bundler emits a 15,324-byte standalone carrier with a 12-byte `__FILE_END__` trailer, without the extra NUL observed in the old embedded section. Structure and payload checks pass, but `mcModuleLoadData` rejects this carrier before numerical execution. Packaging validity, runtime image selection and numerical acceptance are separate checks.

## MCRTC returns a wrapped bitcode buffer in a separate CPU probe

Source `83f6384` independently compiles `extern "C" __global__ void mcrtc_format_probe() {}` through the installed MCRTC API, with zero options and all four visibility masks present and empty. It captures the exact `mcrtcGetBitcodeSize`/`mcrtcGetBitcode` result. This is a new minimal source, not a recompilation or execution of q7. The [producer guide](../experiments/mcrtc_format/README.md) and [CPU observation](../data/inspections/20261008-mcrtc-producer.json) retain the commands, statuses and negative control.

| Observed buffer | Outer form | Total bytes | Bitcode body | Trailing bytes |
| --- | --- | ---: | ---: | --- |
| New MCRTC minimal-source output | LLVM bitcode wrapper, version 0, body offset 20 | 7,360 | 7,332 | 8 zero bytes |
| Earlier q7 extracted bitcode entry | LLVM bitcode wrapper, version 0, body offset 20 | 11,216 | 11,184 | 12 zero bytes |
| Rejected q7 bitcode-only carrier | Clang bundle with empty host and one wrapped-bitcode entry | 15,324 | 11,184 inside its payload | 12-byte `__FILE_END__`, without NUL |

The MCRTC output uses wrapper magic `0x0b17c0de` and raw CPU-type field 255, consistent with the [wrapper structure][bitcode]. That raw field does not identify a physical architecture. Its body starts with LLVM bitcode magic and the installed `llvm-dis` 19.1.3 decodes the **retained output** directly. The emitted module names triple `mxc-metax-macahca`; the probe is `metaxgpu_kernel` with `target-cpu="xcore1000"` and `target-features="+xcore1000"`. These are observations of this source/options/installation, not an ISA decoder or C550 execution qualification. Decoded `source_filename` is `ld-temp.o`; the source lineage comes from the retained API request and output receipt.

MCRTC reports version integers 1/0, distinct from the SDK directory or a compiler build identifier. The valid case has a one-byte NUL log and eight API status records. An independent `#error` case returns compilation status 6 and producer exit 1, preserves its 207-byte log including NUL, and has six status records with no output buffer. Error-string helper calls are additional SDK calls, so these are status-record counts. Both program-destruction calls succeed and both observed CPU process groups are empty after exit. The full source suite passes 241 CPU tests, and 12 pre-API refusals pass on the installed host binary.

This establishes the producer's returned form for one case. It does not establish that the old q7 wrapper loads, that these different sources are byte-identical, or which path the earlier fatbin selected. The earlier carrier refusal and numerical failures remain unchanged. Library-internal device/context behavior was not profiled; the probe itself calls no device enumeration, module loading or kernel launch API.

## Separately measured module-input routes

The successor `60dc2fc` experiment supplies the retained wrappers directly to `mcModuleLoadData`, without repackaging or recompiling device source. It keeps the supplied buffer alive through module unloading. The [module guide](../experiments/wmma_module/README.md) separates the API-only producer stage from q7 collection; [full numerical findings](../wiki/wmma-exactness.md#successor-direct-wrapped-bitcode-loads-and-retains-the-q7-residual) and the [result record](../data/results/20261008-wmma-wrapped-q7.json) retain the measured scope.

| Exact input form | Observed route outcome on C550 / MACA 3.5.3.18 |
| --- | --- |
| 7,360-byte MCRTC producer wrapper | Load, lookup, synchronize and unload succeed. No kernel is launched. |
| 18,232-byte q7 native ELF | Native control completes both launches; the strict q7 contract fails only for WMMA C00. |
| 11,216-byte original q7 wrapper | Both orders and a separate trace complete; all six guarded matrices match their earlier native counterparts after complete input matching. WMMA retains the residual; scalar is exact. |
| 15,324-byte constructed q7 carrier | The earlier `3a08040` attempt remains rejected at loading. It is not retried in the wrapper experiment. |

Each supplied-image receipt compares the complete buffer with its own retained original. These distinct origins are not interchangeable even when their wrapper headers share a convention. All four direct-wrapper processes leave one `.cache` and one `.cache.lock` file in their initially empty requested directories; the native control leaves none. The q7 trace nevertheless reports `is_recompiled=false` for both kernels. Neither file creation nor this event flag identifies cache contents, final instructions or a count of compilation operations. Those questions require separate evidence.

## Retained runtime caches contain distinct native artifacts

A subsequent **CPU-only inspection** examines the four cache snapshots retained by source `60dc2fc`. It creates no new module, compilation or GPU run. The [cache inspection record](../data/inspections/20261008-runtime-cache.json) binds every observation to that completed run and retains image comparisons separately from its unchanged numerical result.

Each cache contains 321 bytes before a three-entry Clang offload bundle. The bundle has an empty host entry, a native ELF labeled `maca-mxc-metax-macahca--xcore1002`, and wrapped bitcode labeled with the same target plus `-bc`. All offsets below are absolute file offsets:

| Retained snapshot | Cache bytes | Native ELF offset / bytes | Wrapped bitcode offset / bytes |
| --- | ---: | --- | --- |
| MCRTC producer load/lookup | 28,173 | 4,417 / 13,912 | 20,801 / 7,360 |
| Each of three q7 wrapper collections | 36,093 | 4,417 / 18,232 | 24,897 / 11,184 |

The bundle descriptor table occupies 202 bytes, ending at cache offset 523. Payload ranges are bounded and nonoverlapping, alignment gaps are zero, and each file ends with the 12-byte `__FILE_END__` marker. Each corresponding `.cache.lock` is empty. Independent parsing agrees with SDK bundler extraction for the representative producer and q7 artifacts. This is an observation of these files; the cache header, filename-key algorithm and lock protocol remain undocumented in the checked sources.

The initial 321 bytes contain a 128-byte fragment matching the beginning of the supplied wrapper, followed by command-like text naming `mxcc` and `--offload-arch=xcore1002`. The fragment is too short to be the declared complete bitcode module. The text has no argument separators and is not a compiler invocation receipt; these fields do not define a supported cache serializer.

### Compare the stored native artifacts

All three q7 caches contain identical native payloads and identical bitcode payloads. Their 18,232-byte native ELF differs from the earlier 18,232-byte ELF extracted from the original executable. Comparisons use the corresponding named symbol ranges, accounting for changed file offsets:

| Native field | Original ELF | Cached ELF |
| --- | ---: | ---: |
| `.text` bytes | 3,400 | 3,656 |
| WMMA symbol offset / bytes | 8,704 / 1,304 | 8,960 / 1,296 |
| Scalar symbol offset / bytes | 10,240 / 1,352 | 10,496 / 1,352 |
| WMMA `.mtreg_count` / `.streg_count` | 28 / 20 | 28 / 21 |
| Scalar `.mtreg_count` / `.streg_count` | 36 / 19 | 36 / 19 |

Both kernel symbol ranges differ in bytes; equal scalar length does not mean identical content. The cached note has a 3,335-byte descriptor, compared with 3,493 bytes in the original. Both kernels retain note fields `.max_block_size=512`, `.kernarg_size_bytes=112`, and zero shared/private memory. The producer cache separately contains an eight-byte `mcrtc_format_probe` symbol and a 1,294-byte note descriptor. Its earlier load-only stage still has no launch or numerical evaluation.

These observations establish different stored native artifacts. They do not identify native instructions, prove which cached bytes ran, or explain the arithmetic residual. Resource-note fields and symbol lengths do not establish hardware ceilings, occupancy or instruction counts.

### Decode the changed cached bitcode

The producer cache's complete 7,360-byte bitcode wrapper equals its original producer output. The q7 cached wrapper differs from its 11,216-byte input: it is 11,184 bytes, with a 20-byte header, 11,152-byte body and 12 zero padding bytes. Installed `llvm-dis` 19.1.3 decodes the retained cached wrapper directly; no source is recompiled.

The cached q7 module keeps triple `mxc-metax-macahca`, the `metaxgpu_kernel` calling convention and each kernel's `target-cpu="xcore1000"` / `target-features="+xcore1000"`. These IR attributes and the bundle's `xcore1002` labels are separately observed fields. The WMMA body removes a duplicate integer loop-bound `and` expression, reducing its static `and` count from 27 to 26. The scalar body shares a B-address OR expression, reducing its static `or` count from 31 to 24. SSA uses and labels change with these edits.

Five static MMA calls remain in WMMA. Scalar retains sixteen FP16-to-FP32 conversions and eight `fmul contract` / `fadd contract` pairs. This selected textual comparison is not an equivalence proof, a compiler-pass trace or a native-arithmetic explanation. The previous q7 exactness failure remains unchanged.

### Reproduce the CPU extraction for these snapshots

For these retained files only, copy bytes from offset 321 through EOF into a fresh `cache.bundle`, after checking the cache size, unique bundle magic and bounded descriptors against the record. The successfully used SDK routes then select the `xcore1002` entries:

```sh
set -eu
SDK_BIN=/opt/maca-3.5.3/mxgpu_llvm/bin
export MACA_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/opt/maca-3.5.3/lib
"$SDK_BIN/clang-offload-bundler" --type=bc --list --input=cache.bundle
"$SDK_BIN/clang-offload-bundler" --type=bc --unbundle --input=cache.bundle \
  --targets=maca-mxc-metax-macahca--xcore1002 --output=cached-device.elf
"$SDK_BIN/clang-offload-bundler" --type=bc --unbundle --input=cache.bundle \
  --targets=maca-mxc-metax-macahca--xcore1002-bc --output=cached.bc
"$SDK_BIN/llvm-dis" cached.bc -o cached.ll
readelf -h -SW -n cached-device.elf
"$SDK_BIN/llvm-nm" --format=posix cached-device.elf
```

Use a fresh output directory and preserve the retained cache. These specimen-specific offsets are not a general cache-format API. The inspection retains 17 successful bounded CPU commands and empty observed process groups after exit. No native decoder, compiler invocation trace, new numerical acceptance or performance result was obtained. A later explicit cached-native experiment would need its own frozen plan and admission.

[bundle]: https://releases.llvm.org/19.1.0/tools/clang/docs/ClangOffloadBundler.html#bundled-binary-file-layout
[bitcode]: https://releases.llvm.org/19.1.0/docs/BitCodeFormat.html#bitcode-wrapper-format
[objcopy]: https://releases.llvm.org/19.1.0/docs/CommandGuide/llvm-objcopy.html
[dis]: https://releases.llvm.org/19.1.0/docs/CommandGuide/llvm-dis.html
[contract]: https://releases.llvm.org/19.1.0/docs/LangRef.html#fast-math-flags
