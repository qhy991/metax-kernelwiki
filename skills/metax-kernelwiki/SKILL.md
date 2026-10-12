---
name: metax-kernelwiki
description: Retrieve evidence-scoped MetaX C550 kernel facts, MACA toolchain guidance and reproducible microbenchmarks from this repository. Use for C550 kernel authoring or performance diagnosis; do not treat C500 documentation or NVIDIA behavior as C550 measurements.
---

# MetaX C550 KernelWiki

Before retrieving authoring material, honor the caller's frozen reference-access and experience/pass assignments. If a treatment withholds measured recipes or complete implementations, do not load them through this skill. Retrieval metadata grants no additional access.

From the repository root run `python3 scripts/wiki.py search <topic>`, then `show <id>`.
For a page that defines a marked summary, use `python3 scripts/wiki.py show <id> --summary`
for its concise view. Omit `--summary` to read the full page, including pages without a summary.
Missing or malformed summary blocks return an error; summaries come from the canonical page.
Read each result's confidence, evidence scope and limitations before applying it.
Follow primary source links for documented APIs and result records for local observations.

For candidate design, search the operation and its mechanism, for example `RMSNorm`,
`execution groups`, `stable routing`, `scan block` or `integer key`. Use
`show c550-conditional-operator-optimization --summary` when that material is allowed.
Keep the reported shape, dtype, semantic bounds, source and baseline with each suggestion.
Imported `source-reported / external-observation` entries are leads from another owner;
they do not become local measurements or default parameters.

For a failure, locate the first failed stage: command entry, source checker, compilation,
module admission, numerical comparison, timing quality or profiling. Read the original
receipt and [compiled-artifact guide](../../docs/compiled-artifacts.md). A missing tool,
a native compiler crash and a wrong output require different next actions. Preserve
unknown causes and original verdicts; do not invent a result store to classify them.

For a new experiment, read `AGENTS.md`, `docs/methodology.md` and the installed
gpu-infra lease lifecycle. Use the node's existing allocator and a committed probe.
Store raw evidence outside source. Do not install this skill or launch tests merely to retrieve a fact.

If there is no matching local evidence, report the gap and propose a bounded discriminating experiment.
