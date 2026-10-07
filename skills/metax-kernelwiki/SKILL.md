---
name: metax-kernelwiki
description: Retrieve evidence-scoped MetaX C550 kernel facts, MACA toolchain guidance and reproducible microbenchmarks from this repository. Use for C550 kernel authoring or performance diagnosis; do not treat C500 documentation or NVIDIA behavior as C550 measurements.
---

# MetaX C550 KernelWiki

From the repository root run `python3 scripts/wiki.py search <topic>`, then `show <id>`.
Read each result's confidence, evidence scope and limitations before applying it.
Follow primary source links for documented APIs and result records for local observations.

For a new experiment, read `AGENTS.md`, `docs/methodology.md` and the installed
gpu-infra lease lifecycle. Use the node's existing allocator and a committed probe.
Store raw evidence outside source. Do not install this skill or launch tests merely to retrieve a fact.

If there is no matching local evidence, report the gap and propose a bounded discriminating experiment.
