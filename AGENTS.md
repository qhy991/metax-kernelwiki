# MetaX KernelWiki authoring

- This independent repository owns C550 kernel knowledge and probes. It does not own open-cake-ir Target declarations, GPU allocation or deployment policy.
- Read `docs/methodology.md` before experiments and the current `gpu-infra` GPU lease lifecycle before any device work. Reuse the node's existing allocator. Do not replace a live broker or interfere with other jobs.
- Keep `data/catalog.json` the single retrieval index. Put explanations in `wiki/`, source guidance in `docs/`, bounded raw-result projections in `data/results/`, and reproducible probes in `experiments/`.
- Write public documentation, catalog titles, tags, limitations, and new explanatory result text in English. Preserve original source URLs, identifiers, commands, and retained evidence. Keep one canonical English page per topic.
- Scope each statement to its product, SDK/compiler, workload and observation. Cite primary sources. Mark unmeasured hypotheses explicitly. A source mentioning C500 is not a C550 measurement.
- Prepare and compile without a GPU lease. Commit probe changes before executing a frozen checkout. Save raw evidence outside the source checkout under a new run id; never overwrite an earlier failure.
- Check full outputs with an independent oracle and negative controls before interpreting timings. Preserve timing interval, cache state, warmup, samples, clock/occupancy limitations and selected device identity.
- Follow one question through hypothesis, experiment, evidence, bounded conclusion, next question. Promote no instruction fact from latency alone.
- Inspect worktrees before editing; use `task/metax-<subject>` in an isolated worktree. Preserve unrelated branches and active checkouts. No routine digest inventories.
- Validate `python3 scripts/wiki.py validate` and relevant probe host tests. Record exact commits and commands; do not claim GPU coverage from CPU tests.
