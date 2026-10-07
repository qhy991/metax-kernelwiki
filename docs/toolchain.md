# C550 工具链：原生目标与兼容接口

本页区分网上源码、官方文档和本地安装。第一方入口及适用范围见 [资料索引](sources.md)。C550 的测量结果以实验记录为准。

## 当前环境记录边界

2026-10-07，C550-1 的只读检查报告宿主安装目录为 `/opt/maca-3.5.3`，MXCC 版本输出为 `1.0.0 (6477545d4d)`。这两个版本标识各自保留：前者是 SDK 路径，后者是编译器报告，不能相互替代。最终复现记录还需保存驱动、运行时、容器和 Python 包版本，以及实际调用的编译器绝对路径。

本地头文件检查还发现 `mc_runtime_api.h` 定义了 `waveSize` 到 `warpSize` 的兼容宏，而类型声明中出现两个名称。官方运行时指南的查询例子用 `waveSize`。因此“两个拼写都存在”不等于两个独立硬件事实；设备属性探针应同时保留所用源码、包含路径和输出。此处不填写尚未收录到实验记录的属性值。

## 三条开发入口

| 入口 | 作用与可验证材料 |
| --- | --- |
| MXCC / MXMACA C++ | 直接使用 MACA 运行时和 kernel 扩展；保存原生目标、完整命令、生成的设备代码及运行错误 |
| cu-bridge / cucc | 供 CUDA 风格源代码适配 MACA；官方教程给出 `/opt/maca/tools/cu-bridge/bin/cucc`。保存实际路径与展开后的编译命令 |
| mcTriton | Python/Triton 前端经 MetaX backend 生成设备代码；保存 wheel 版本、backend 文件位置、target、编译参数与产物 |

原生入口来自[运行时指南 3.5.3.x][runtime]，兼容入口来自[官方向量加教程][vector-guide]。初轮使用的[旧运行时指南入口][runtime-legacy]在 2026-10-07 的版本选择器中标为 3.0.0.x；本页已改用重新查阅的 3.5.3.x 固定版本。本页的 mcTriton 实现观察固定在[源码版本 `7dd407c`][triton-readme]；尚未核对它与 C550 的已装包是否相同。

## 原生编译模板

官方文档的基础形式为 `mxcc -x maca`，并显式指定 MACA 路径。下面是待结合实机已验证目标使用的模板，不是已通过的测试记录：

```sh
export MACA_PATH=/opt/maca-3.5.3
export LD_LIBRARY_PATH="$MACA_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
: "${C550_ARCH:?请先从本机设备与编译证据设置原生架构}"
"$MACA_PATH/mxgpu_llvm/bin/mxcc" \
  -x maca -O3 --offload-arch="$C550_ARCH" \
  --maca-path="$MACA_PATH" probe.cpp -o probe
```

`--offload-arch` 的参数拼写与支持范围以本地 `mxcc --help` 和成功编译日志为准。网上[发布说明][release-changes]将 `-offload-arch=native` 列在 3.7.0 的变化中；因此不把它作为宿主 3.5.3 的默认命令。原生目标不能从 `torch.cuda.get_device_capability()` 或 Triton 的 `arch=80` 推导。

## 512-thread 函数属性与运行时重编译

[运行时指南 3.5.3.x §4.3][runtime-cache]给出一个未标注 `__launch_bounds__`、以每 block 1024 线程执行的 vector-add 示例，并说明该例因 block 超过 512 而触发重编译和 binary cache。这支持在本机测试“默认编译产物与较大 block 的运行时路径是否不同”，不支持把 trace 中的 `max_block_size=512` 直接解释为设备的最大线程数，也不保证 1024 线程更快。

[同版本 §4.4][runtime-env]定义 `MACA_CACHE_PATH` 指定二进制缓存目录，默认位置为用户目录下的 `.metax/shadercache/`；`MACA_CACHE_DISABLE=1` 禁用缓存，设为 0 或不设置则启用。实验可为新运行指定独立目录，保留首次执行前后的目录清单、函数属性、完整输出与 host 计时，再在同一缓存目录启动后续进程。目录变化与首发延迟是重编译假说的线索；要确认发生了编译，还需运行时日志或产物等直接证据。此处的 binary cache 是编译缓存，不能当作 L2/HBM 缓存控制。

本轮公开文档检索尚未找到适用于当前 MXCC 的 `__launch_bounds__` 完整语义定义，尤其是第二参数的处理方式和超过显式第一参数时的行为。保留本机头文件、编译诊断和受控运行证据后再作结论，不直接套用 CUDA 或 HIP 的定义。

## mcTriton 源码事实

[Python driver][triton-driver] 在 target 中保留 backend 名 `maca`，并使用 64-lane 组；launcher 将 `num_warps` 乘以 64 作为 block 的线程数。[C driver][triton-driver-c]另有兼容 capability 映射：设备 `major=10/15/16` 分别映射为 `80/86/89`。这些是该源码版本的接口实现，不能当作 NVIDIA compute capability 或 C550 原生 ISA 型号。

[compiler.py][triton-compiler]的阶段是：

```text
Triton → TTIR → TTGIR → LLVM 方言 MLIR → LLVM IR → mcfatbin
```

源码允许 `num_warps` 为 1、2、4、8、16，默认值为 4；声明 `num_stages` 默认值 3，并有 `basic` 和 `cpasync` 等 pipeline 路径。选项通过 Python 检查只说明其被软件接受；形状、类型和具体 pipeline 的可编译性、正确性及性能仍需分别测试。首次实验先用默认配置建立可复现基线。

[triton_metax.cc][triton-codegen]调用 MXCC 的 `--fatbin` 路线，带有 `-maca-link -input-is-device`，并链接 MACA bitcode 库。可用的源码调试开关包括：

```sh
TRITON_PRINT_COMPILE_OPTIONS=1
TRITON_COMPILER_DUMP_ALL=1
```

第一个开关打印编译命令，第二个在 MXCC 命令中加入 `--keep`。它们只有在本机 backend 保留相应实现时才有效。启用时将缓存及中间产物放在当前实验目录，避免把旧目标的缓存误记为新编译结果。网上 README 的构建流程需要另取 `metax_llvm`；本页不要求替换现有 SDK 或重建已有 wheel。

## 计时与 profiler 的证据范围

MACA 提供事件计时 API，官方例子将开始/结束事件放在 kernel 两侧，等待结束事件后查询间隔；[vLLM-metax 的实际包装][kernel-timer]也使用这条路线。实验须说明是否批量执行、预热次数、同步位置、输入准备和输出传输是否在计时内，并保留每次样本。短 kernel 的事件均摊值不能直接解释为纯指令延迟。

缓存重置需要独立说明。没有验证过重置方法时，结果应标注“缓存状态未控制”或实际采用的热复用条件，不能声称冷 L2 或 HBM 测量。发布说明中的驱动优化项也不能代替当前进程的缓存状态证据。

[mcProfiler 手册][profiler]提供性能计数器采集入口；[mxvs 手册][mxvs]覆盖设备、链路、内存和算力工具。当前资料核查没有验证 C550 所装 profiler 的 CLI、指标、权限或采集结果。后续应保存工具版本、原始帮助、采集命令、指标定义和成功输出；工具不可用时明确记录覆盖缺口，保留正确性及计时结果。

[mcTracer 3.5.3.x 采集章节][tracer]说明工具导出 JSON，[Viewer 章节][tracer-viewer]说明由专用 UI 打开。已查阅这两个章节，未找到 JSON `ts` / `dur` 字段的单位契约。安装的 MCPTI 头文件对上游时间戳的注释不能证明 exporter 没有转换单位；在取得导出实现或明确契约前，保留原始数值与“单位未验证”标记。已采集 trace 的范围与验收见 [profiling 页面](../wiki/profiling.md)。

[runtime]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/index.html
[runtime-legacy]: https://developer.metax-tech.com/api/client/document/preview/567/C500_RuntimeAPIProgrammingGuide_CN.html
[runtime-cache]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编译和调试.html#binary-cache
[runtime-env]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编译和调试.html#irs9wigbg1oh1
[vector-guide]: https://gitee.com/metax-maca/mxmaca-performance-tuning-guide/blob/main/guide/ch1.初探异构编程.vectoradd.md
[release-changes]: https://developer.metax-tech.com/api/client/document/preview/发布说明/MXMACA_发布说明/曦云C500系列/latest/split_files/新增特性及变更.html
[triton-readme]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/README.md
[triton-driver]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.py
[triton-driver-c]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.c
[triton-compiler]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/compiler.py
[triton-codegen]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/triton_metax.cc
[kernel-timer]: https://github.com/MetaX-MACA/vLLM-metax/blob/f2fcc59c314f1fbc7897f46d465e5f7c35900e8a/csrc/libtorch_stable/quantization/awq/hgemv_selector.hpp
[profiler]: https://developer.metax-tech.com/api/client/document/file/211/preview/?file_type=pdf
[tracer]: https://developer.metax-tech.com/api/client/document/preview/性能测试及分析工具/mcTracer使用手册/曦云C500系列/3.5.3.x/split_files/mctracer.html
[tracer-viewer]: https://developer.metax-tech.com/api/client/document/preview/性能测试及分析工具/mcTracer使用手册/曦云C500系列/3.5.3.x/split_files/mctracer_viewer.html
[mxvs]: https://developer.metax-tech.com/api/client/document/preview/996/index.html
