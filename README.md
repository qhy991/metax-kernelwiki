# MetaX KernelWiki

An evidence-based kernel engineering wiki for **MetaX C550**. It connects MACA documentation, reproducible native probes, measured results, and the limits of each finding.

The wiki is written in **English**. Original source URLs retain their published spelling. Its lightweight retrieval follows [metal-kernelwiki](https://github.com/qhy991/metal-kernelwiki); its mechanism-and-evidence approach follows [bw1100-kernelwiki](https://github.com/qhy991/bw1100-kernelwiki).

## Start here

- **Writing a C550 kernel?** Read [device identity](wiki/device-identity.md) and the [MACA toolchain guide](docs/toolchain.md), then search for the operation or mechanism you need.
- **Evaluating an optimization?** Read the [measurement contract](docs/methodology.md) and the relevant finding below. Check its inputs, software version, correctness result, and limitations.
- **Following the research?** The [research log](docs/research-log.md) records completed experiments and open questions. The [source index](docs/sources.md) separates official documentation from upstream code and local evidence.

## Measured findings

The published device experiments use one **C550 with MACA 3.5.3.18**. Each page links to its frozen probe source and result records. A result applies to the stated device, software, inputs, and protocol.

| Topic | What the evidence supports | Read more |
| --- | --- | --- |
| Device identity | The runtime, native ISA, compiler family, and compatibility architecture describe different interfaces. | [Identity and resource observations](wiki/device-identity.md) |
| Global memory | Copy configurations have repeated observations; fixed-address read permutations isolate one source of variation while changing the output permutation. | [Copy and stride](wiki/memory-access.md) · [Read order](wiki/memory-order.md) |
| Transpose | Tiling, shared-memory row pitch, and dynamic shared-memory requests have controlled comparisons. The request-size response depends on shape. | [Tiling and shared memory](wiki/transpose.md) |
| Wave collectives | Full typed masks select 64- or 32-element reduction groups in the tested SDK. Physical wave width remains 64. | [Shuffle, reduction, and mask types](wiki/wave-collectives.md) |
| Launch bounds | A function attribute of 512 did not prevent the tested 1024-thread launch; explicit bounds changed the observed recompilation path. | [Launch bounds and runtime recompilation](wiki/launch-bounds.md) |
| WMMA exactness | At q6/q7/q12, tested exponent redistribution and factor-sign transfers preserve complete outputs at fixed products: q6 pairs are exact; q7/q12 retain -2^-31. Components and scalar controls pass; WMMA performance remains unaccepted. | [Numerical diagnosis and controls](wiki/wmma-exactness.md) |
| Profiling | A zero tool exit code does not establish a usable GPU trace. Kernel events and resource fields need separate checks; exported time units remain unverified. | [Trace acceptance](wiki/profiling.md) |

These are bounded findings, not general hardware guarantees. In particular, documentation for another C500-series product does not establish a C550 measurement, and a microbenchmark gain does not establish an end-to-end gain.

## Search locally

Retrieval uses only the Python standard library and requires no GPU or network:

```sh
python3 scripts/wiki.py list
python3 scripts/wiki.py search wave
python3 scripts/wiki.py show c550-wmma-exactness
python3 scripts/wiki.py show c550-wmma-exactness --summary
python3 scripts/wiki.py validate
```

The WMMA page provides a concise `--summary` view drawn from its own Current findings section. Full `show` remains available for every page.

[`data/catalog.json`](data/catalog.json) is the single retrieval index. Each entry names its confidence, evidence scope, sources, and limitations. The command-line tool reads the same pages linked above.

## Reproduce an experiment

Each probe guide defines preparation, compilation, device execution, and independent output checking:

| Probe | Guide |
| --- | --- |
| Device properties, copy, stride, and launch bounds | [Native probes](experiments/native/README.md) |
| Fixed-address read permutations | [Memory-order probe](experiments/memory_order/README.md) |
| Transpose, row pitch, and shared-memory requests | [Transpose controls](experiments/transpose/README.md) |
| Shuffle and integer reduction | [Wave collectives](experiments/wave_collectives/README.md) |
| FP16 inputs, FP32 outputs, and scalar controls | [WMMA diagnostics](experiments/wmma/README.md) |

Read the [methodology](docs/methodology.md) before running a probe. Prepare and compile before acquiring a GPU; use the node's existing allocator. Retain device outputs, end the device worker, verify release, then perform host analysis. GPU work follows the [gpu-infra lease lifecycle](https://github.com/qhy991/gpu-infra/blob/main/skills/gpu-infra/SKILL.md#gpu-lease-lifecycle).

## Evidence and contributions

- Keep full-output correctness checks ahead of performance acceptance. A failed diagnostic remains a failure; its timings do not become accepted performance data.
- Record the timer, warmup, repeated samples, cache-state treatment, device identity, and concurrency limitations. Logical bandwidth is not measured DRAM bandwidth.
- Preserve raw logs, failures, and negative controls outside the source checkout under distinct run IDs. Publish bounded projections in [`data/results/`](data/results/).
- Commit probe changes before execution. A new question or changed contract uses a successor source and a new run; it does not overwrite earlier evidence.
- Follow [AGENTS.md](AGENTS.md) when extending the wiki. See [provenance](docs/provenance.md) for repository discovery and reference boundaries.

This repository is independent of open-cake-ir. Its findings do not automatically change Compiler Targets, calibrations, or qualification sets.
