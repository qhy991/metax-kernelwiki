# Observe the installed MCRTC producer's output

[Home](../../README.md) · [Toolchain](../../docs/toolchain.md) · [Earlier carrier refusal](../../wiki/wmma-exactness.md#successor-bitcode-only-carrier-rejected-before-execution)

This probe captures the exact buffer returned by the installed MCRTC compiler API for an independently authored empty kernel. It addresses the producer's serialization, before any decision about a new module-loading experiment. It does not compile the q7 source, load a module or launch a kernel.

The installed `/opt/maca-3.5.3/include/mcr/mcrtc.h` declares `mcrtcGetBitcodeSize` and `mcrtcGetBitcode`. The public symbols are exported by `libmcruntime.so`; no separate `libmcrtc` link is assumed. The [3.5.3 guide](https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编译和调试.html#vfjw0l1pd0sk1) uses different `GetCode` names in its example, so the installed declarations own this probe's spelling.

## Fixed calls and two independent cases

The valid source is:

```cpp
extern "C" __global__ void mcrtc_format_probe() {}
```

The negative case prepends `#error MCRTC_FORMAT_NEGATIVE_CONTROL`. Both use the same program name, `mcrtc_format_probe.mc`, zero headers and **zero compile options with a null option pointer**. That zero-option call is allowed by the installed header and used by the installed non-cooperative-groups sample. No architecture option is supplied; an omitted compiler default is not evidence of a C550 target.

Each process queries `mcrtcVersion`, creates the program and compiles it. It queries and copies the complete compiler log even after compilation failure. After successful compilation only, it queries and copies the complete `GetBitcode` output. Every returned program handle gets one destruction attempt, including after an API or file-retention failure. The collector records each API's numeric status, enum name and error string before proceeding.

The literal source, options, program name and visibility masks are retained in `raw.jsonl`. `compile.log` and `producer-output.bin` preserve exactly the API-reported byte extents, including NUL bytes. A zero-length API output remains zero length; it is not rewritten into an invented artifact. Each buffer has a recorded 64 MiB capture cap. This is a bounded serialization probe, not a general compiler harness.

## Compile the host collector

Use a fresh output path with an existing parent directory. The host executable uses only the MCRTC header and standard C++ headers, linked against the installed runtime library:

```sh
MACA_PATH=/opt/maca-3.5.3 \
MACA_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
bash experiments/mcrtc_format/compile.sh /absolute/fresh/build/producer
```

Retain the host compiler version and complete command. The host's `-std=c++17` flag is separate from the **zero** options passed to MCRTC. Freeze the source before running the API probe. No GPU lease is requested for this host compiler/API observation.

## Capture and check

Create a fresh empty output directory for each case, then run each bounded CPU child separately:

```sh
MACA_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
timeout --signal=TERM --kill-after=10s 120s \
  /absolute/frozen/build/producer --capture valid /absolute/fresh/valid-output
```

Use `--capture compile-error` with another fresh directory for the independent negative case. The collector refuses missing or nonempty visibility masks, invalid case names, and missing/nonempty directories before its first MCRTC call. It calls no device-enumeration, module-load or launch API. The library's internal device/context behavior is not profiled, and no guarantee about those internals is inferred from the source-level call list.

Producer exit 0 means the required successful-producer API calls and byte retention completed. Exit 1 preserves `MCRTC_ERROR_COMPILATION` with its log and program cleanup. Other API, retention or cleanup errors return 2. Keep every failed attempt under its original directory; changing options creates a separate experiment.

The standard-library checker binds a receipt to its declared case:

```sh
python3 experiments/mcrtc_format/check.py /absolute/retained/valid-output --case valid
python3 experiments/mcrtc_format/check.py /absolute/retained/error-output --case compile-error
```

Checker exit 0 means the receipt matches the requested case. For the negative control, the JSON still reports the producer's `compile-failed` status and exit 1. It requires the unique diagnostic marker, successful log capture/destruction and no output-payload success. Wrong byte extents, incomplete records, nonempty masks and cleanup failures cannot pass. This check establishes neither module loadability nor numerical correctness.

## Inspect bytes without recompilation

Identify the returned buffer from its actual bytes. If it has a known wrapper, bundle or ELF structure, report the bounded fields and tool results. If its format is unknown, retain that observation. Use the installed decoder on the retained buffer where supported; do not substitute a new `-emit-llvm` source compilation or silently repackage it.

Even if this producer returns a format resembling the earlier q7 payload, it is a different source and artifact. Format resemblance establishes no byte identity, loader route, JIT mechanism, historical fatbin selection, numerical acceptance or performance result. The earlier generated-carrier refusal remains unchanged.
