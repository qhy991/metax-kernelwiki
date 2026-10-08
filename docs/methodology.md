# Experiment and knowledge-maintenance methods

[Home](../README.md)

This repository develops testable optimization knowledge for C550. Each page should identify the cost to reduce, the concrete rewrite, semantic constraints, conditions for use, possible regressions, and the evidence available. Record sources before drawing conclusions. Even when the device model is unchanged, retain software versions, inputs, and the measurement contract.

## Evidence categories

Record knowledge confidence separately from experimental coverage. A label does not replace the conditions under which a statement holds.

| Knowledge label | Meaning |
|---|---|
| `documented` | A mechanism or API contract explicitly stated by a primary source |
| `source-reported` | An observation in source code, an author's account, or an issue report; it does not automatically become a local fact |
| `inferred` | An explanation derived from existing facts; state its basis and alternatives still to exclude |
| `experimental` | An unverified design, hypothesis, or candidate |
| `locally-measured` | A retained local measurement with an explicit device, software stack, input, and contract |

Use the same `evidence_scope` values supported by the [catalog](../data/catalog.json) and [query tool](../scripts/wiki.py):

| Scope | Meaning |
|---|---|
| `external-observation` | An observation or environment check from another record, not yet a measurement in this repository |
| `upstream-source` | Content supported by upstream documentation or source code |
| `protocol` | An experimental design or measurement contract that does not yet constitute a device result |
| `compile-only` | Compilation evidence, with no execution-correctness or speed claim |
| `device-correctness` | A device-correctness result for specified inputs and a specified contract |
| `local-measurement` | A local measurement with a result record, interpreted within that record's conditions |

Environment and compilation records do not support speed claims. A paired comparison must identify its baseline, both source versions, common inputs, precision, and timing interval. A bounded pass is not a general hardware guarantee, and a component improvement is not an end-to-end improvement. Record failures and superseded states in the original record's explanation rather than creating another scope enumeration.

Keep the retrieval date, source publication date, and software version separate. Mark references to mutable `main`, `master`, or official web pages as mutable. Prefer the fixed commit actually inspected; do not guess a version. Distinguish compatibility names in software from physical hardware identity.

## What to freeze before each run

1. State one testable question and its comparison, such as how batched device time for contiguous copy varies with data size. Define success, failure, and stopping conditions first.
2. Record the device name and available identity properties, driver, runtime, compiler version, compilation command, source commit, process, and device selection. Use `unknown` for undetermined fields.
3. Fix shape, dtype, layout/stride, allocation method, input generation, seed, external oracle, tolerance, and timing boundaries. A changed contract requires a new run.
4. Follow the installed `gpu-infra` skill's allocation and release procedure for the backend. Record the actual owner, selected device, shared lock or lease, and visible-process checks. Under cooperative `local_serialized` allocation, claim only that participating broker jobs serialize through the device lock. An empty process snapshot does not establish whole-machine exclusivity; do not invent an exclusive-allocation receipt.
5. Complete compilation and host-side contract checks before bounded device execution. The probe retains raw timings and final outputs. Run the complete CPU oracle after the device process exits, and accept no timing result before correctness passes. Preserve errors, exceptions, timeouts, and failures as observed. A retry creates a new record and never overwrites the failure.

Keep raw run directories outside the source repository and installed skills. Each run must retain at least the executed source, commands, environment, stdout/stderr, elementwise correctness checks, and raw timing samples. Documentation changes need no GPU allocation. Rerun the relevant validation only when semantics or timing implementation changes.

## Scope of the initial native probes

| Probe | Question | Required boundary |
|---|---|---|
| Device properties and minimal compile/run | Which properties does this runtime report, and can the compiler generate and execute target code? | Separate API observations from official hardware guarantees; do not default missing properties |
| Contiguous copy | How does complete-copy kernel batch time vary with working set and block configuration? | Check tails; distinguish logical read/write volume from actual hardware transactions |
| Strided copy | How does the specified read/write stride affect this probe's access efficiency? | State stride units, allocation span, valid element count, output layout, and bounds checks |
| Empty/minimal kernel batches | What are the average device batch time and host submission cost for small kernels? | An event interval alone cannot measure pure host launch overhead; report the two times separately |

These are experimental designs, not completed results. They do not establish support for GEMM, matrix instructions, atomics, reductions, communication, complete models, or every dtype. Bind the device name and MACA version to each run's environment record. The first observed version is not a permanent baseline.

## Timing contract

Start with one fixed stream; the initial native probe uses the default stream. Create and initialize events, allocate memory, copy inputs, compile, and warm up before timing. Retain the MACA API and version used. Interpret timing through that version's API documentation and the actual probe.

For device batch timing, enqueue a start event, a fixed number of kernel launches, and an end event in order on the same stream. Wait for the end event before reading elapsed time. Save every raw batch duration and repetition count. `batch duration / kernel count` is the mean time per launch within that contract. It may include stream gaps, insufficient host submission rate, and event-boundary overhead; do not call it isolated kernel execution time or pure launch latency.

Use a separate host monotonic clock for the required boundary. A clock around the submission loop measures host enqueue time; one that includes completion synchronization measures host completion time. Before comparing synchronous and asynchronous interfaces, state whether computation is complete when each returns. Do not combine host, event, and profiler times into one metric.

Separate warmups from measured samples. Interleave baseline/candidate order or randomize it in advance and retain the order. Use independent processes when needed to examine process-to-process variation. Report batch count, launches per batch, all raw samples, and median [min, max]. Repeated kernels within one batch are not independent samples. Do not present p95 from very few batches as statistically representative.

Record device-state handling before each sample. No explicit application cache flush does not prove that the runtime performs none. MACA launch mode can affect cache policy: retain the values or unset state of `MACA_LAUNCH_MODE`, `MACA_LAUNCH_BLOCKING`, and `MACA_DIRECT_DISPATCH`, and label the runtime policy unknown until checked. A large working set alone does not establish cold-cache conditions. Repeated kernels on the same input measure a particular reuse condition. Record available clock, power, temperature, and concurrency observations; retain limitations when clocks are not locked or exclusivity is unproven. Allocator cleanup is not evidence of a GPU cache flush.

Keep later profiler captures separate from unprofiled timings. Successful tool startup, file creation, parseable trace data, valid kernel events, and usable counters are separate checks. Without parsed evidence, report collection status only and do not infer a bottleneck.

## Numerical correctness and byte counts

Check every valid copy output, including small inputs, incomplete blocks, and the studied strides. Inputs should distinguish positions so constant data cannot hide addressing errors. Use exact comparison for exact-copy semantics. Define the contract before testing NaN bit patterns or aliasing, and use suitable bitwise checks. Allocate from the maximum accessed index, not just the number of valid elements. Retain guard/sentinel checks and state untested aliasing behavior.

Compute effective copy bandwidth from a predefined logical byte count. If each valid element is read once and written once, `N` elements of `s` bytes imply `2*N*s` logical bytes, with decimal `GB/s = 2*N*s / seconds / 1e9`. Report the actual allocation span separately for strided cases; holes are not useful traffic. This metric is not measured DRAM bandwidth and does not establish cache hit rate, bus transactions, or utilization of an advertised peak.

Before extending to reductions, approximate functions, or matrix arithmetic, fix intermediate accumulation precision, rounding, error metrics, and the external oracle. Equal output dtype does not establish equal computation contracts. Retain incorrect or slower inputs before discussing conditions of use; do not hide failures behind dispatch predicates.

## From results to knowledge pages

A result must identify the device/version, question, shape/dtype/layout, baseline and candidate, correctness, timing interval, raw samples, state handling, profiler coverage, artifact references, and uncovered scope. Negative results can be useful knowledge. An unsupported parameter recommendation does not become a default through experience alone.

Write mechanism pages as short causal explanations: the original cost, what the rewrite changes, its preconditions, possible new costs, and the extent of local support. Keep precise values and full configurations in the owning experiment record. The page should retain the observations that determine applicability and link to their sources. Update the existing page for a mechanism rather than adding empty pages to increase a count.

Preserve refuted observations in their original records. Explain why they became invalid or were superseded, and identify the successor run. Distinguish tool defects, input errors, compilation limits, performance hypotheses, and hardware capabilities. Without a direct comparison, do not claim that a choice is faster or best.

## Local maintenance checks

Run `python3 scripts/wiki.py validate` before committing. The current validator checks catalog schema, unique IDs, required fields, evidence labels, and page paths. `locally-measured` means an observation came from the local device; its result must exist and contain run, source, device, environment, correctness, measurement, and limitation fields. The two scopes have separate admission rules:

- `local-measurement` retains the strict performance-evidence gate: full-output `correctness.passed` must be the Boolean `true`. Diagnostic labels cannot make a failed result pass this gate.
- `device-correctness` can record directly measured correctness passes or failures. The result's own `evidence_scope` must also be `device-correctness`; `correctness.passed` must be an actual Boolean; and a nonempty `correctness.tested_contract` must state the tested contract. It also requires `measurement.purpose="correctness_diagnostic"` and `measurement.performance_accepted=false`. Its timing records are not accepted as performance conclusions.

Device outputs that violate the original contract can therefore become locally measured negative knowledge. The original result remains failed, with its oracle and tolerance unchanged. Classifying a diagnostic record for retrieval does not establish numerical acceptance, authorize a tolerance change, or permit performance claims from the failed run.

Exercise `search` and `show` for affected entries, for example `python3 scripts/wiki.py search timing` and `python3 scripts/wiki.py show maca-event-timing`. A maintainer must check source support, agreement between prose and results, accidental speed claims from compilation records, and missing baselines in paired comparisons. Do not describe these manual checks as implemented validator capabilities.

Structural validation does not open or verify raw experiments outside the repository. It does not prove performance validity, continuing custody, exclusive resources, or formal qualification. Metadata-only changes do not trigger GPU jobs or provide additional publication authorization. Do not compute or list digests routinely; locate evidence by run ID, path, and commit.
