# MACA traces: process success does not establish capture success

[Home](../README.md) · [Catalog](../data/catalog.json) · [Probe guide](../experiments/native/README.md)

The installed `mcTracer 3.5.3.18-ef9e10e` independently profiled the initial copy with block size 256. Event timing from profiled runs is kept separate from unprofiled measurements; it does not replace the original performance samples.

## An output-directory failure can still return zero

Run `20261007-trace-01` passed an absolute path to `--odname`. This version appended that path to the working directory, failed to create the directory and logged `FATAL`, yet returned exit code 0. The application still produced output. Neither application output nor the process exit code alone establishes successful profiling.

Successor `20261007-trace-02` used relative `--odname trace` in a fresh run directory. It produced parseable JSON and passed the complete CPU output check for that copy. The first failure was retained, not rewritten as a success.

## A version query can also return zero without a version

During the [native-module q7 experiment](wmma-exactness.md#successor-explicit-native-elf-module-loading), `/opt/maca/bin/mcTracer --version` returned zero with `execvpe: No such file or directory` and only startup/end messages. That attempt established no tool build version. Its later actual capture was checked separately for the expected two kernel events and process release. The trace's `process_name` metadata contained `C500` and `version: 0.0.1`; those fields do not identify the tracer product build. Preserve the failed query and use observed output content, not exit status alone, when recording tool identity.

## What the trace actually covers

The earlier copy capture, `20261007-trace-02`, contains 5171 events, including **1020 GPU copy-kernel events** corresponding to 20 warmups and 1000 timed launches. Their correlation IDs match the host `mcLaunchKernel` calls one to one. Host API counts alone are not GPU dispatch counts.

All 1020 kernel events report block 256, grid 16385, 6 registers per thread, zero static/dynamic shared memory and zero private memory. These are the tool's reports for this artifact, not a general register limit or occupancy calibration. `max_block_size=512` appears only in some events; missing values must not become zero. This field is distinct from the device runtime's maximum of 1024 threads.

The trace metadata label `C500` is a tool label. The exact native runtime device name remained `MetaX C550`; a compatibility label does not change the target identity.

## Trace time units remain unverified

The JSON does not declare a time unit. Installed MCPTI headers describe activity timestamps as ns, and the exported timestamp magnitude and event-batch span are consistent with ns. That is still insufficient to establish that the exporter applies no transformation. This wiki retains raw `dur` values with units marked unverified, not as calibrated kernel latency in microseconds.

After removing 20 warmups, the 1000 GPU events have raw `dur` median 48640 and range [46848, 52224]. They come from the profiled execution. See the [capture result](../data/results/20261007-trace-02.json) for the complete analysis.

The exporter's unit contract or implementation must be checked independently before comparing kernel intervals with event intervals from the same capture. This trace supplies no validated cache/DRAM performance counters, so it does not identify a memory bottleneck.

## Successor investigation of 512/1024-thread launches

The [launch-bounds experiment](launch-bounds.md) confirmed that a default function can execute correctly with 1024 threads while its runtime attribute remains 512. Execution-variant trace fields and explicit function declarations must be recorded separately. The added `--initial-launches 1` option separates the initial diagnostic launch from the following 20 warmups. The summarizer removes one contiguous prefix and must therefore be used on a single-case trace; the total warmup count of a mixed sweep is not one such prefix.
