# C550: distinguish product, ISA and compiler identities

[Home](../README.md) · [Catalog](../data/catalog.json)

One product can have several architecture names. Each answers a different question:

| Field | Meaning | How to verify it |
| --- | --- | --- |
| Product name | The physical board, such as MetaX C550 | `mx-smi` and the native runtime |
| Device ISA | The instruction target reported by the device | `macainfo` |
| Native codegen family | The target family selected by the installed compiler | Compiler flags, generated artifacts and compiler tools |
| MACA capability | The runtime's major/minor values | A property-query program compiled against the installed SDK |
| Triton architecture | The backend API value used to select a compilation route | Driver/compiler source and actual metadata from the installed version |

An earlier [open-cake-ir investigation](https://github.com/qhy991/open-cake-ir/blob/main/docs/metax-c550-bringup.md) observed `XCORE1002`, `xcore1000` and native capability `10.2` on C550. These are historical external observations. This wiki independently queried and executed C550 in [20261007-native-01](../data/results/20261007-native-01.json), using MACA 3.5.3.18 and driver 3.6.11. It records the device ISA and compiler family separately.

## Why wave width matters

Consecutive threads accessing consecutive elements can enable coalesced transactions. Data exchange across a wave cannot simply reuse shuffle or ballot code that assumes 32 lanes. The official series documentation and the [inspected MACA Triton backend source](../docs/toolchain.md) declare 64 lanes; device observations are recorded separately.

When porting a CUDA kernel, check whether constants such as `32`, `0xffffffff`, `lane & 31` and `threadIdx.x >> 5` encode hardware assumptions. An algorithmic tile may contain 32 elements without making the hardware execution group 32 lanes wide.

**Questions for device experiments:** Do blocks that are not whole waves work correctly? Do block sizes 64/128/256/512 change latency at the same useful access volume? Device properties alone cannot answer these questions; they require checked kernel outputs and timing.

## Header aliases are not independent observations

Inspection of the MACA 3.5.3 headers found `#define waveSize warpSize` in `mc_runtime_api.h`, while the type declarations contain both names. Printing both names can therefore read the same field. Compile property queries against the installed SDK and check every return code. Equal values from these aliases are not two independent hardware confirmations.

## Runtime observations on 2026-10-07

The device run required exactly one visible card with the exact runtime name `MetaX C550`. It reported a wave width of 64, 104 multiprocessors, a maximum of 1024 threads per block and 2048 threads per multiprocessor, 65536 bytes of shared memory per block, and 8388608 bytes of L2. These are API observations, not occupancy calibration. The [result record](../data/results/20261007-native-01.json) retains the raw fields.

`mx-smi` reported 65536 MiB of board memory. In the same experiment, `mcDeviceProp_t.totalGlobalMem` returned 68283269120 bytes, or 63.59375 GiB. Both observations are retained separately. Neither establishes that 64 GiB can be allocated, and the maximum allocatable capacity was not measured.
