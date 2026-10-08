# Explicit native and wrapped-bitcode module checks

[Home](../../README.md) · [Artifact inspection](../../docs/compiled-artifacts.md) · [Original q7 collector](../wmma_q7/README.md)

This host-only collector keeps the supplied module format explicit. It contains no device kernel source. Native ELF and retained q7 bitcode forms use one numerical collection path. The independently generated MCRTC image has a separate load/lookup mode that never launches its empty kernel. Every failed route remains a failure; another input kind is never selected automatically.

The current q7 protocol is **v3**. Earlier native-module v1 results replay at [`b463205`](https://github.com/qhy991/metax-kernelwiki/blob/b463205eff646cd0c88fe26fd51838be1f8f1726/experiments/wmma_module/README.md); the rejected-carrier v2 experiment replays at [`3a08040`](https://github.com/qhy991/metax-kernelwiki/blob/3a0804098493b2bc750097e0ca6720a2d2449780/experiments/wmma_module/README.md). The current checker does not translate those raw records. A separate five-record producer-load mode has its own schema and no numerical result.

## Bounded image forms

| Required image kind | Examined form | Input boundary |
| --- | --- | --- |
| `native-elf` | 18,232-byte ELF64 little-endian, ET_DYN, raw machine 253 | The extracted native payload from q7 source `9b7bef6` |
| `retained-bitcode-bundle` | 15,324-byte Clang offload bundle made by the installed MetaX tool | The earlier rejected carrier; retained support does not make it a fallback |
| `retained-wrapped-bitcode` | Direct 11,216-byte LLVM wrapper: offset 20, body 11,184, 12 zero tail bytes | The unchanged extracted q7 bitcode, with no added carrier |
| `mcrtc-producer-wrapped` | Direct 7,360-byte LLVM wrapper: offset 20, body 7,332, 8 zero tail bytes | The independently generated minimal-source artifact; only `--load-producer`, never q7 collection |

The bitcode carrier has two descriptors: `host-x86_64-unknown-linux-gnu` at offset 4,096 with size zero, and `maca-mxc-metax-macahca--xcore1000-bc` at offset 4,096 with size 11,216. Its descriptor table ends at byte 145, followed by zero padding. The payload has a 20-byte wrapper, 11,184 bytes of bitcode and 12 zero padding bytes. The carrier ends with the 12-byte `__FILE_END__` marker **without NUL**. This differs from the old embedded fatbin's 13-byte tail and is the actual standalone bundler output.

These are closed checks for four retained forms, not a general ELF or bitcode verifier. The bare inner bitcode body, a mixed native/bitcode bundle, an extra entry, a changed layout or another SDK output is outside this protocol. `xcore1000` remains the compiled family; it does not rename the physical C550 target.

The collector retains the complete supplied buffer as `loaded-image.bin` and passes that buffer to `mcModuleLoadData`. In q7 mode it resolves the original WMMA/scalar symbols and uses the same collection and launch code for each admitted q7 kind. Four argument addresses represent A, B, output payload and unsigned chunk count 1; `extra` is null. The runtime owns packing and hidden arguments. The [3.5.3 API warning](https://developer.metax-tech.com/api/client/document/preview/990/split_files/mxmaca_运行时api模块.html) conflicts with installed and [published mcTriton examples](https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.py); v1 already measured this argument-array convention for the native image. Each new route still requires its own load, lookup, launch and correctness evidence.

## Prepare without a GPU lease

The current direct-wrapper plan reads both original retained wrappers unchanged; it performs no rebundling or device-source recompilation. The earlier carrier's construction commands and refusal remain in the [frozen v2 guide](https://github.com/qhy991/metax-kernelwiki/blob/3a0804098493b2bc750097e0ca6720a2d2449780/experiments/wmma_module/README.md).

Compile the host driver without a GPU lease:

```sh
MACA_PATH=/opt/maca-3.5.3 CXX=/usr/bin/g++ LD_LIBRARY_PATH=/opt/maca-3.5.3/lib \
MACA_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
bash experiments/wmma_module/compile.sh /absolute/fresh/build/module-repro
```

Retain the frozen source commit, host compiler version and complete build command. CPU masks belong only to these children; the allocator owns the later device mapping. Read the [methodology](../../docs/methodology.md) and [GPU lease lifecycle](https://github.com/qhy991/gpu-infra/blob/main/skills/gpu-infra/SKILL.md#gpu-lease-lifecycle). Module loading may involve runtime compilation internally; keep that device-context-dependent operation in the bounded worker instead of claiming it was completed during packaging.

Direct wrappers are taken unchanged from their independent retained sources. CPU admission can inspect either direct form before any device work:

```sh
python3 experiments/wmma_module/check.py --inspect-image /absolute/retained/q7-embedded.bc \
  --image-kind retained-wrapped-bitcode
python3 experiments/wmma_module/check.py --inspect-image /absolute/retained/producer-output.bin \
  --image-kind mcrtc-producer-wrapped
```

These two commands check the closed wrapper structures. The post-run check separately compares the whole supplied image with its original reference.

## Load the independent producer image first

The [MCRTC producer](../mcrtc_format/README.md#observed-result) generated and retained its 7,360-byte buffer in an earlier process, then destroyed its compiler program. The new load mode keeps that exact buffer alive through lookup and module cleanup. The installed sample frees its buffer after loading; this probe does not depend on that early-release behavior.

Within a fresh admitted C550 worker, with a fresh empty output directory:

```sh
/absolute/frozen/build/module-repro --load-producer \
  /absolute/retained/producer-output.bin /absolute/fresh/producer-load-output
```

It records request/environment, exact device identity, successful `mcModuleLoadData`, lookup of `mcrtc_format_probe`, synchronization and unload. It creates no q7 input/output arrays and calls no kernel-launch API. A successful receipt qualifies these API operations for this saved artifact only. Library-internal work, native instructions and execution remain outside that claim.

After process and allocator release:

```sh
python3 experiments/wmma_module/check.py /absolute/retained/producer-load-output \
  --producer-load --expected-image /absolute/independently-retained/producer-output.bin
```

The five-record checker requires exact image equality, fixed symbol, synchronization and unload, and reports no numerical pass. A failure stops the planned later stages. If loading succeeded before another error, the owner attempts synchronization and unload once and records cleanup failures separately; it does not retry an invalidated handle. Successful completion is written only after cleanup succeeds.

## Collect under the existing allocator

The planned native control is a separate declared run, never a fallback. Use a fresh empty output directory for each collection. Inside an admitted worker:

```sh
/absolute/frozen/build/module-repro --run wmma-first retained-wrapped-bitcode \
  /absolute/retained/q7-embedded.bc /absolute/fresh/bitcode-output
```

For the separately declared native control, select `native-elf` with its retained ELF and another fresh output directory. Use `scalar-first` for the opposite bitcode order, under a new admitted worker. Any API or integrity failure stops later stages; it does not select raw bitcode or another carrier automatically.

For the new direct-wrapper experiment, select `retained-wrapped-bitcode` and the original `q7-embedded.bc` file in its own run. The older carrier remains an independently recorded refusal; it is not retried in the direct-wrapper plan. The predeclared sequence is producer load/lookup, native q7 control, q7 wrapper in both orders, and a separate wrapper resource trace, each with a fresh admission and post-release check.

All q7 collections preserve the [fixed q7 numerical contract](../wmma_q7/README.md): 1,024 halfwords per operand, one padded K16 step, 256 FP32 outputs, 64 guards per side and complete input snapshots before/between/after implementations. WMMA uses one 64-thread block and scalar one 256-thread block. The same A/B allocations are uploaded once, and outputs are separately initialized and retained. There are no warmups or collector event timers. The exact oracle remains -1/256 at C00 and zero elsewhere; the observed residual is never an expected value.

Protocol v3 binds the image kind in module, protocol and completion records. Its environment fields include the actual value or unset state of `MACA_MODULE_LOADING`. An error names the failing API expression. Successful collection unloads the module and frees buffers; verify process and allocator release before CPU checking. A trace is a separate diagnostic run, with exporter units unverified.

## Check after release

For the direct q7 wrapper, require its independent original image reference:

```sh
python3 experiments/wmma_module/check.py /absolute/retained/bitcode-output \
  --image-kind retained-wrapped-bitcode \
  --expected-image /absolute/independently-retained/q7-embedded.bc
```

For native mode, use `--image-kind native-elf` and its independent ELF as `--expected-image`. For direct q7 bitcode, use `--image-kind retained-wrapped-bitcode` and the original extracted wrapper as `--expected-image`. Omit `--expected-bitcode` for both direct forms: the entire image is already the independently referenced artifact. The checker compares all supplied image bytes and validates the selected form. The retained carrier mode additionally requires its independent original wrapped-bitcode reference; it is outside the new direct-wrapper run plan. It rejects self/hard-link reference comparisons and inconsistent route metadata. Numerical/file checks share the original q7 oracle, while each module protocol is checked separately.

The q7 checker exits 0 for complete exact success, 1 for a complete numerical failure, and 2 for an integrity or structural failure. The producer-load checker reports only its API/image receipt and has no numerical pass. Actual image-input identity does not prove unchanged final instructions, a cold cache, a specific compilation mechanism, the old fatbin's selected entry or the arithmetic cause. Preserve failures and partial metadata as their own outcomes. A successful load is separate from numerical acceptance and performance acceptance.

## Retained v2 outcome

The [source `3a08040` result](../../wiki/wmma-exactness.md#successor-bitcode-only-carrier-rejected-before-execution) completes the native control with the earlier numerical residual. Its first bitcode-carrier attempt is rejected by `mcModuleLoadData` with `mcErrorNoKernelImageForDevice`, before function lookup, input upload or launch. Both later stages remain unstarted. The failed attempt's actual supplied buffer passes post-release identity/structure checks, but this carrier attempt produces no numerical result. Keep the raw API failure and partial records; the complete-output checker cannot accept a run that never reaches its nine-record protocol.

## Retained v3 outcome

The [source `60dc2fc` result](../../wiki/wmma-exactness.md#successor-direct-wrapped-bitcode-loads-and-retains-the-q7-residual) completes all five declared stages. The MCRTC producer image passes load/lookup/synchronize/unload without a launch or numerical evaluation. The native q7 control and three direct-wrapper q7 collections each fail exact acceptance only at WMMA C00, with residual -2^-31; scalar is exact. Complete inputs, readbacks and guards pass. Both wrapper orders and the separate trace preserve the earlier native output buffers.

This qualifies the supplied forms for these operations on the recorded runtime. It does not explain the carrier refusal, earlier fatbin selection or final native arithmetic. The trace has two kernel events; its units remain unverified and no performance is accepted. Use the [public record](../../data/results/20261008-wmma-wrapped-q7.json) for complete output words, API receipts and resource observations.
