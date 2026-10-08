# Provenance and reuse boundaries

[Home](../README.md) · [Methodology](methodology.md)

This wiki provides searchable knowledge for MetaX C550 kernel development and optimization. Hardware identity, MACA APIs, compiler behavior, and performance claims each need evidence for this target. Borrowing another kernel wiki's organization does not establish C550 hardware facts.

## Reference repositories inspected

The references below were inspected on **2026-10-07** at the listed revisions. This is the access date, not a source publication date.

| Reference | Inspected revision | Organizational ideas used here |
| --- | --- | --- |
| [qhy991/bw1100-kernelwiki](https://github.com/qhy991/bw1100-kernelwiki) | `main`, [`6b4a6de8cdece72fea1ac8dd8fee0a104e491a94`](https://github.com/qhy991/bw1100-kernelwiki/tree/6b4a6de8cdece72fea1ac8dd8fee0a104e491a94) | Record sources first. Explain each mechanism through its cost, rewrite, preconditions, possible regressions, and local evidence. Use `evidence_scope` to distinguish environment, compilation, numerical checks, paired timing, and invalid experiments. |
| [qhy991/metal-kernelwiki](https://github.com/qhy991/metal-kernelwiki) | `main`, [`30d36da6c1d2543372793cbf020863a5f26dce85`](https://github.com/qhy991/metal-kernelwiki/tree/30d36da6c1d2543372793cbf020863a5f26dce85) | Use one page/source catalog, offline retrieval, and a thin skill entry point. Separate public knowledge from raw experiments outside the repository. State the boundaries between documentation, inference, and local measurement. |

The BW1100 repository is private, so its links may require access. The Metal repository was public when inspected. Its provenance page cites an earlier BW1100 commit, `52ae9a1`; the table above records the revision inspected for this wiki. These are separate inspections.

Specific references:

- [BW1100 maintenance rules](https://github.com/qhy991/bw1100-kernelwiki/blob/6b4a6de8cdece72fea1ac8dd8fee0a104e491a94/MAINTENANCE.md) require causal explanations, applicability conditions, and counterexamples, while preserving corrected observations.
- [BW1100 evidence validation](https://github.com/qhy991/bw1100-kernelwiki/blob/6b4a6de8cdece72fea1ac8dd8fee0a104e491a94/scripts/validate-evidence.py) distinguishes evidence scopes. Compilation and protocol records cannot support performance claims; paired comparisons need a denominator.
- [BW1100 knowledge tests](https://github.com/qhy991/bw1100-kernelwiki/blob/6b4a6de8cdece72fea1ac8dd8fee0a104e491a94/tests/test_knowledge.py) check retrieval, filtering, citations, and evidence constraints. Structural tests are not GPU validation.
- [Metal maintenance rules](https://github.com/qhy991/metal-kernelwiki/blob/30d36da6c1d2543372793cbf020863a5f26dce85/MAINTENANCE.md) use logical `artifact_ref` values for experiments outside the repository and explain that a page alone cannot let an external reader replay the original run.
- [Metal measurement methods](https://github.com/qhy991/metal-kernelwiki/blob/30d36da6c1d2543372793cbf020863a5f26dce85/wiki/measurement.md) separate timing scope, correctness, raw samples, and profiler observations. Allocator cleanup does not establish that hardware caches were flushed.
- [Metal provenance](https://github.com/qhy991/metal-kernelwiki/blob/30d36da6c1d2543372793cbf020863a5f26dce85/PROVENANCE.md) preserves reuse and licensing boundaries. A citation does not grant permission to redistribute the full source content.

## Evidence owned by this wiki

Upstream documentation and source links support only what they state. A documented contract, an API return value, a successful compilation, and a device execution are different kinds of evidence. This wiki does not import another vendor's wave/warp width, cache properties, instructions, timers, or resource limits as MetaX facts. A compatibility API name does not establish identical underlying hardware.

Each local result must identify its run, executed source, environment, inputs, oracle, failures, raw timing samples, and measurement scope. Experiment directories outside the source repository own the raw records; wiki pages hold derived explanations and references. Public pages use logical `run-id/relative-file` references. Machine addresses, accounts, keys, and private path mappings stay outside public content. If raw records are not distributed with the repository, state that limitation rather than claiming that third parties can already replay the original run independently.

The reference inspections did not rerun experiments on the reference repositories' devices. This repository did not copy BW1100 or Apple experimental results, knowledge-page text, or retrieval implementations, and did not use their GPU measurements to set C550 defaults. Any later direct code reuse must identify the files, fixed revision, and actual license separately. A citation cannot supply a missing third-party license.

See the [methodology](methodology.md) for this wiki's measurement and maintenance rules. Source count is not coverage: new conclusions need new, traceable evidence, and successful retries must not overwrite earlier failures.
