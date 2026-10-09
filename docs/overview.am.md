---
title: MetaX KernelWiki for kernel agents
subtitle: What the skill contains, how an agent reads it, and what its evidence supports
lang: en
template: sheet
theme: blueprint
cols: 2
source: Repository snapshot 4ce41c9 · 2026-10-09
---

The skill helps an agent find C550 mechanisms and check their conditions before changing a kernel. The repository retains both successful observations and failed numerical contracts.

## A The skill and its knowledge {span=2}

```callout info One entry point, one retrieval index
`skills/metax-kernelwiki/SKILL.md` tells the agent how to retrieve and interpret knowledge. `data/catalog.json` indexes the canonical pages. The Python CLI reads those pages without a GPU or network.
```

| Location | What it contains |
| --- | --- |
| `skills/metax-kernelwiki/SKILL.md` | Retrieval instructions and the rule for missing evidence |
| `data/catalog.json` | 10 entries with confidence, evidence scope, sources and limitations |
| `wiki/` | Explanations of device behavior and measured mechanisms |
| `docs/` | Toolchain, methodology, artifact inspection and research history |
| `experiments/` | Reproducible probes with 8 probe guides |
| `data/results/` and `data/inspections/` | Published result projections and artifact inspections |

Counts describe snapshot `4ce41c9`. Full raw experiments remain outside the source checkout. [Skill source](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/skills/metax-kernelwiki/SKILL.md) · [Catalog](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/data/catalog.json)

## B How the agent uses it

```flow
Task -> Search: operation or symptom
Search -> Page: matching topic
Page -> Evidence: scope and limitations
Evidence -> Candidate: applicable observation
Candidate -> Device_check: validate this workload
Evidence -> New_probe: missing local evidence
New_probe -> Knowledge: retained result and bounded conclusion
```

Retrieval itself starts no experiment. A new probe needs a fixed question, an oracle and the existing device allocator. The skill does not install itself while answering a query.

## C Commands the agent can use

```sh
python3 scripts/wiki.py list
python3 scripts/wiki.py search wave
python3 scripts/wiki.py show c550-wave64-collectives
python3 scripts/wiki.py show c550-wmma-exactness --summary
python3 scripts/wiki.py validate
```

Run these commands from the repository root. `--summary` reads a marked section of the canonical page. A missing or malformed summary returns an error. Use full `show` for any page.

[Query implementation](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/scripts/wiki.py)

## D The ten indexed topics {span=2}

| Topic | Question it helps answer | Condition to retain |
| --- | --- | --- |
| Device identity | Which names describe the board, ISA and compiler route? | Compatibility architecture is a software interface |
| MACA toolchain | How do MACA C++ and mcTriton produce artifacts? | Compilation, loading and execution are separate observations |
| Event timing | What interval does a sample measure? | Batch intervals can include submission gaps |
| Copy and stride | Which access configurations have local evidence? | Logical bandwidth is not measured DRAM traffic |
| Profiling | Is a trace usable for diagnosis? | Check kernel events and the meaning of each field |
| Launch bounds | What does a launch attribute constrain here? | Retain compiler version and runtime recompilation evidence |
| Read order | How does a fixed-address permutation affect this probe? | The output permutation and short-term reuse can change |
| Transpose | How do tiling, pitch and shared requests interact? | Shape and code generation affect the outcome |
| Wave collectives | Which threads participate in shuffle or reduction? | Physical width, subgroup width and mask type differ |
| WMMA exactness | Does the matrix path meet the specified numerical contract? | Failed exactness cannot support an accepted speed claim |

[Topic index and evidence labels](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/data/catalog.json)

## E Three examples of useful knowledge {span=2}

| Example | Retained observation | How the agent should use it |
| --- | --- | --- |
| Wave reduction | Physical wave width is 64. With input 1..64, the tested full 64-bit mask gives 2080. The full 32-bit interface gives 528 and 1552 in the two halves. | Preserve the mask argument type and check the intended group. Sparse masks and floating-point reduction need their own validation. |
| Transpose | Tiling improves the tested direct-access implementation. Later controls separate row pitch from declared storage capacity and dynamic requests. | Treat pitch and shared allocation as distinct choices. Recheck the target shape. Timing alone does not identify bank conflicts. |
| WMMA q7 | The exact result is `0xbb800000`. The tested WMMA path gives `0xbb800001`, while the scalar control is exact. | Keep the numerical failure. Check the workload's oracle before accepting matrix-path performance. The arithmetic cause remains unresolved. |

Both type and full-mask value change in the wave comparison. It is not a type-only experiment. These examples come from one C550 with MACA 3.5.3.18.

[Wave evidence](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/wiki/wave-collectives.md) · [Transpose controls](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/wiki/transpose.md) · [WMMA findings](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/wiki/wmma-exactness.md#current-findings)

## F Reading an evidence label

| Field | Meaning | Example |
| --- | --- | --- |
| `confidence` | Where the knowledge comes from | `documented`, `inferred`, `locally-measured` |
| `evidence_scope` | What the record examined | `compile-only`, `device-correctness`, `local-measurement` |
| `limitations` | Where the conclusion stops | One SDK, complete masks, or fixed inputs |
| `sources` and `result` | Where to inspect support | Primary document or retained result projection |

`locally-measured` can describe a correctness failure. Read the result's acceptance fields. The current catalog has 7 local measurements, 1 device-correctness entry, 1 upstream-source entry and 1 protocol entry.

[Evidence definitions](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/docs/methodology.md#evidence-categories)

## G How it relates to open-cake-ir

KernelWiki provides mechanisms, counterexamples and hypotheses. An agent can use them to propose a hardware-specific Schedule or a separately admitted native experiment.

CAKE checks its modeled contracts and evaluates the candidate on the exact target. A reusable change needs its own Compiler or verifier review and successor validation.

The wiki owns no Target declaration, calibration or GPU allocator. A wiki observation does not automatically grant a CAKE capability. A controlled comparison must declare which knowledge each author can read.

[Repository boundaries](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/AGENTS.md) · [Knowledge maintenance](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/docs/methodology.md#from-results-to-knowledge-pages)

## H Current boundaries and next questions {span=2}

- Published device findings use one C550 and MACA 3.5.3.18. A different SDK needs fresh evidence.
- Event batches, host clocks and profiler durations describe different intervals. Raw profiler time units remain unverified.
- Cached native bytes differ from the original artifact. This does not identify the instructions that actually ran.
- The WMMA residual's arithmetic cause remains unresolved. Its diagnostic timings are not accepted performance results.
- Sparse masks, floating-point collective behavior and broader framework coverage remain open questions.
- A new experiment fixes its contract and source first. It retains failed outputs and checks device release before host analysis.

This guide summarizes the cited snapshot. The catalog and canonical pages remain the knowledge source. [Research log](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/docs/research-log.md) · [Artifact inspection](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/docs/compiled-artifacts.md) · [Measurement method](https://github.com/qhy991/metax-kernelwiki/blob/4ce41c9/docs/methodology.md)
