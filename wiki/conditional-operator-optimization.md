# Conditional operator optimization from external C550 studies

[Home](../README.md) · [Method](../docs/methodology.md) · [Catalog](../data/catalog.json)

These reports come from open-cake-ir, not new experiments in this repository. The retrieval label is `source-reported / external-observation`. Read the owning report for the exact source, inputs, timing and result. Its initial CAKE implementation is the performance baseline; Torch owns correctness and is not the reported speedup denominator.

<!-- kernelwiki:summary:start -->
## What to try and when

- For the reported BF16 residual RMSNorm shape `[1,131,8192]`, eight execution groups are a supported candidate relative to the initial four-group CAKE mapping. Preserve cast points and test cache hints separately.
- For the reported INT32 stable-routing shape `[4,1024,8]`, a larger scan block and a bounded integer key are separate candidate mechanisms. The source does not establish an extra benefit from pipeline stage two.
- Key `e*N+p` requires `0 <= e < 256`, `0 <= p < N`, and `N <= 2**23`, plus the original stability semantics. INT32 storage alone supplies none of these bounds.
- Match baseline, original numerical contract, timing interval and version before reusing a result. Timed samples are not independent Agent searches.
- Distinguish a transport or compile failure from an incorrect or slower executed kernel. A protocol fault label does not establish the underlying cause.

These are conditional search suggestions. They establish no default for another shape, strong expert-library advantage, Compiler causality or cross-hardware effect.
<!-- kernelwiki:summary:end -->

## Execution-group allocation in RMSNorm

The [RMS factor study](https://github.com/qhy991/open-cake-ir/blob/c35a0f8f308eb2cc34f87f1e1becd98033daaf3d/docs/results/metax/RMS_FACTOR_ABLATION_20261011.md) reports C550/MACA3.5.3, public Compiler source `3d28af7b` and Bench `ababa4c0`. This version string does not establish that it is identical to this wiki's MACA3.5.3.18 environment.

The mechanism changes work distribution within a row while retaining the arithmetic, tile, grid, ABI and BF16 cast locations. The report tests execution-group allocation and streamed cache hints as separate factors. It attributes the main observed improvement to the allocation change; the isolated cache-hint comparison did not pass its predefined improvement threshold.

Try the allocation as a candidate for the same contract. Test other row lengths, row counts, resource commitments and numerical contracts afresh. More threads can increase total block resources even when the reported register count per thread falls. The report has no achieved-occupancy, bandwidth or instruction-count measurement that identifies a hardware bottleneck. It does not promote eight groups to a default or add a Compiler pass.

## Scan tiling and stable integer keys

The [stable-routing study](https://github.com/qhy991/open-cake-ir/blob/c35a0f8f308eb2cc34f87f1e1becd98033daaf3d/docs/results/metax/STABLE_ROUTING_SEARCH_20261011.md) compares a fixed initial CAKE implementation with explicit candidates under public source `f3546b36` and Bench `ababa4c0`.

Increasing scan rows from128 to512 changes work partitioning. It must preserve complete coverage, unique output stores and tails. The underlying scans remain quadratic; the report establishes no complexity improvement or general superiority to an expert implementation.

Encoding expert index `e` and original assignment position `p` as `e*N+p` can replace a lexicographic comparison. Under the stated bounds, the largest key is `256*N-1 <= 2**31-1`. Distinct expert indices preserve ordering, and assignment position retains the required stable order. Outside the semantic range, refuse the rewrite or use a verified fallback. Do not infer range from dtype or ABI.

The factorial comparisons retain block size, pipeline level and key encoding separately. They do not show an additional benefit from stage two. Separate baseline comparisons do not estimate a precise factor interaction. Read the report's raw scope before attributing small differences.

## Combine row grouping with execution-group choices as a hypothesis

Grouping independent rows reduces the number of blocks and increases each block's work. Changing execution groups also changes the mapping and possible reduction tree. Evaluate these choices together without changing epsilon, arithmetic precision, cast boundaries, inputs or the external oracle.

For a Cake candidate, the caller's Compiler owns legality and route admission. A source or wiki recipe cannot open an undeclared operation. Use its existing transformation APIs, preserve refusals and qualify each complete candidate. Native compilation alone supplies no numerical or speed result. The reported wide-row native failures are investigation leads, not evidence for a universal register limit or a row-grouping prohibition.

## Diagnose before scoring

The [experiment audit](https://github.com/qhy991/open-cake-ir/blob/c35a0f8f308eb2cc34f87f1e1becd98033daaf3d/docs/results/metax/EXPERIMENT_AUDIT_20261012.md) separates wrong device binding, author time allocation, reduction lowering, source generation and receipt faults. It retains the historical outcomes.

Inspect the first failed stage and its original command, candidate, native output or comparison. A source checker missing from a device image is tool availability, not evidence that the kernel contains an undefined name. A segfault in a native compiler's register-allocation stack identifies that stage but does not prove register-capacity exhaustion. Keep an unexplained timeout unexplained.

For tolerant comparisons, prove that an unwritten-output sentinel lies outside every expected element's acceptance interval. A tiny finite sentinel may pass a zero-output check. Check original distributions and inputs; retain raw NaN/Inf output before a JSON conversion can hide the numerical failure. Byte-exact copying and complete arithmetic qualification have different coverage.

Respect the caller's frozen policy for failures, budgets and repeated measurements. Metadata or a new recipe cannot reclassify a sealed failure or authorize repeated sampling until a result passes. This repository supplies retrieval and probes, not the caller's Run, allocation or Compiler admission owner.

## Source and applicability ledger

| Source | Supported use | Missing scope |
|---|---|---|
| Pinned RMS factor report | A conditional execution-group candidate and separate cache-factor observation | Other shapes, an experience treatment, expert-library comparison |
| Pinned stable-routing report | Scan-block and bounded-key candidates with controlled factors | Unbounded values, asymptotic improvement, precise factor interaction |
| Pinned experiment audit | Failure-stage and baseline interpretation for the reported workflow | Root causes it explicitly leaves unknown; current caller qualification |

The source reports retain the exact numerical values. This page adds no copied results table, new local result, GPU call, timing or promotion. Its source links are pinned; citations within those reports retain their own version rules.
