# q7 through an explicitly supplied native module

[Home](../../README.md) · [Retained artifact inspection](../../docs/compiled-artifacts.md) · [Original q7 collector](../wmma_q7/README.md)

This successor asks whether the native ELF extracted from the retained q7 executable can be loaded through MACA's module API, and whether that route reproduces the numerical result. The host driver contains no device kernel source. It loads only the supplied native image; it never supplies the surrounding fatbin or its bitcode entry.

A successful collection establishes the behavior of this explicit module route. It does not reveal which payload the earlier fatbin selected, establish that the loader makes no transformations, identify native instruction precision, or qualify a new compiler. The original exact oracle and failed records remain unchanged.

## Artifact and argument boundary

Use the 18,232-byte native device ELF extracted from source `9b7bef6`'s retained q7 host executable, as described in the [inspection guide](../../docs/compiled-artifacts.md). This diagnostic restricts input to the examined ELF64 little-endian, ET_DYN, raw-machine-253 form. Its header check is not a general ELF verifier or an instruction decoder.

The collector reads the image into host memory, retains those exact bytes as `loaded-image.elf`, and passes that buffer to `mcModuleLoadData`. It resolves the original mangled WMMA/scalar symbols. No device-source compilation or bitcode input occurs in this route. The post-run checker requires an independent `--expected-image` file and compares every byte with the retained supplied image. This checks the live module-input handoff; it does not establish the final runtime instruction bytes.

The installed 3.5.3 header and [official API documentation](https://developer.metax-tech.com/api/client/document/preview/990/split_files/mxmaca_运行时api模块.html) warn that `kernelParams` is unimplemented and describe `extra`. However, the installed `vectorAdd_mcrtc` sample and pinned published [MetaX mcTriton launcher](https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.py) pass a `kernelParams` array with a null `extra`. This collector tests that concrete sample pattern. Documentary disagreement is not evidence of runtime support or failure.

Four arguments are passed by address: A pointer, B pointer, output-payload pointer and unsigned chunk count 1. The runtime owns argument packing and hidden parameters. The collector does not construct a 112-byte buffer from the ELF note or infer an ABI from its hidden-argument descriptions. A failed API call is retained as a route failure at that stage.

## Preserved numerical contract

The fixed inputs, oracle and intended observation protocol follow the [standalone q7 contract](../wmma_q7/README.md): two nonzero terms, full padded K16, 1,024 halfwords per operand, 256 FP32 outputs, 64 guards per side and complete input snapshots before/between/after implementations. WMMA uses one 64-thread block; scalar uses one 256-thread block. Both use the same uploaded A/B allocations and separate initialized output buffers.

There is one launch per implementation, no warmup and no event timing. Exact acceptance compares all outputs against independent Fraction arithmetic, with -1/256 at C00 and zero elsewhere. The previously observed residual is never an expected value. Opposite orders are separate collections. Device execution ends before CPU checking.

## Prepare and compile without a GPU lease

The module ELF must come from an independently retained extraction. Keep it outside the source checkout. The new host executable is compiled with a host C++ compiler and links the installed MACA runtime:

```sh
MACA_PATH=/opt/maca-3.5.3 \
MACA_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
bash experiments/wmma_module/compile.sh /absolute/fresh/build/module-repro
```

Retain the source commit, compiler version, full command and diagnostics. Commit the source before execution. Read the [methodology](../../docs/methodology.md) and [GPU lease lifecycle](https://github.com/qhy991/gpu-infra/blob/main/skills/gpu-infra/SKILL.md#gpu-lease-lifecycle). Compilation masks belong only to that CPU child; the allocator owns device mapping for the later worker.

## Collect under the existing allocator

Create a fresh empty output directory. Run this command only within an admitted device worker:

```sh
/absolute/frozen/build/module-repro --run wmma-first \
  /absolute/retained/embedded-device.elf /absolute/fresh/output
```

Use `scalar-first` for the opposite order. The command refuses invalid arguments, a nonempty output directory or the wrong basic image form before querying a device. Runtime admission requires exactly one visible device, named MetaX C550 with wave size 64. After successful collection, the module is unloaded and buffers are freed. Verify worker/process and allocator release independently of its completion record.

After release, run the CPU-only checker:

```sh
python3 experiments/wmma_module/check.py /absolute/retained/output \
  --expected-image /absolute/independently-retained/embedded-device.elf
```

The checker retains exact failure as exit 1, integrity or structural failure as exit 2, and complete exact success as exit 0. Its arithmetic/file checks share the original q7 oracle; its module protocol is validated separately. Do not rewrite a module record into the old collector schema. A trace, if collected separately, is diagnostic resource evidence with unverified exporter time units, not accepted performance.

## Retained records

Nine ordered JSONL records describe device, supplied module, protocol, before snapshot, first implementation, between snapshot, second implementation, after snapshot and completion. Module admission and exact symbol names are explicit. Variant records include the four-argument interface, launch geometry and observed module-function attributes. The supplied image, prepared inputs, six input snapshots and both guarded output buffers remain outside the source checkout with the command and allocation/release receipts.

Source `b463205` has a [published device result](../../wiki/wmma-exactness.md#successor-explicit-native-elf-module-loading): both kernels load and launch in three paired collections, but the exact numerical contract remains failed because WMMA retains one C00 residual in each collection. Scalar outputs are exact. API success is separate from numerical acceptance.
