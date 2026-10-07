# 第一方资料索引

查阅日期：2026-10-07。此页是资料入口，不是 C550 测量结果。实机证据应保存设备型号、原生架构、软件版本、命令和原始输出，并链接到对应实验记录。

## 适用范围

MetaX 的 **MXC500 系列**是软件文档的适配集合，包含 C500、C500X、C550、C550-PL、C588、N260 和 X206；集合成员不因此拥有相同容量、带宽、缓存或计算能力。官方发布说明同时区分 MXC600 系列与 MXC600-U。引用 C500 微架构教程时保留 C500 标签，待 C550 验证后再形成结论。[发布说明：概述][release-overview]

本轮未找到足以单独证明 **C550 → xcore1002** 映射的公开官方产品规格页。该映射、内存容量、处理器数量和原生 ISA 应由本次 C550 的设备查询及编译产物确认，不从 C500 文档或 CUDA 兼容值推导。

## 已核实的文档入口

| 资料 | 可以支持什么 | 使用边界 |
| --- | --- | --- |
| [曦云系列运行时 API 编程指南][runtime] | 执行模型称 64 个线程为 wave；设备查询、内存、事件、kernel 启动和 MXCC 工程构建的入口 | 系列指南；数值仍需 C550 查询。指南示例用 `mcDeviceProp_t.waveSize`，本地 SDK 头文件可能含兼容别名 |
| [MXMACA 发布说明：概述][release-overview] | 各软件组件的发布版本和产品系列适用范围 | `latest` 会移动，不等于实机安装版本 |
| [MXMACA 发布说明：新增特性及变更][release-changes] | 判断功能在什么版本出现；例如 3.7.0 节记录 `-offload-arch=native`，也记录工具与编译器变更 | 不据此认定宿主 3.5.3 已具备新版本功能；也不据旧记录否定本地回移补丁 |
| [MXMACA 发布说明：已知问题和使用限制][release-limits] | 发现需针对性排查的版本及场景限制，包括 C550 OAM 的部分通信算子问题 | 限制有具体场景，不能外推到所有 C550 kernel；不直接照抄环境变量作为优化配置 |
| [官方性能优化指南 README][guide] | `guide/`、`case/`、`microbenchmark/` 的导航；原仓库声明测试设备为 C500 与 A100 | 其 C500 数值只作为待验证的先验。检索时页面未声明许可证，优先链接、转述及独立编写探针 |
| [官方向量加教程][vector-guide] | 可参考 `cucc` 路由、设备内存使用、预热、事件计时和错误检查 | 教程属于 C500 场景；用 CUDA 名称的接口是兼容表面，不是 NVIDIA 硬件身份 |
| [mcProfiler 使用手册 PDF][profiler] | 性能计数器采集工具的使用入口、任务参数和指标选择 | 链接是较早文档；本次尚未验证 C550 所装工具的 CLI、权限和指标集 |
| [mxvs 测试工具套件 3.5.3.x 目录][mxvs] | 设备信息、PCIe、Memory、MetaXLink 和算力测试的官方参考入口 | 与自写 kernel 的计时范围不同，结果需分别报告；本轮未运行 |

## 已核实的官方代码

下列 mcTriton 链接固定在 `7dd407c26568fceaca44cb894138e5202d369805`，其分支名是 `3.0`。该版本是网上代码观察，不代表 C550 本机 wheel 的源码身份。

| 文件 | 可审查的具体实现 |
| --- | --- |
| [mcTriton README][triton-readme] | 构建依赖来自 MACA 软件栈与 `metax_llvm`；构建脚本为 `maca_tools/build_triton.sh` |
| [backend/driver.py][triton-driver] | 返回 `maca` backend 与 64-lane target；launcher 使用 `64 * num_warps` 个线程 |
| [backend/driver.c][triton-driver-c] | 将设备 `major` 映射为 Triton capability：10→80、15→86、16→89；资源查询调用 MACA API |
| [backend/compiler.py][triton-compiler] | `mcfatbin` 输出、编译阶段、MACA pipeline 选项以及 `num_warps` 检查 |
| [triton_metax.cc][triton-codegen] | LLVM IR 经 MXCC 形成 fatbin；包含打印编译命令和保留中间产物的开关 |
| [vLLM-metax 事件计时包装][kernel-timer] | 固定在 `f2fcc59c314f1fbc7897f46d465e5f7c35900e8a`；直接使用 `mc_runtime.h` 与 `mcEventElapsedTime` 的实际代码例子 |
| [mcTVM][mctvm] | 官方另一条编译路线的入口；README 的 `metax/mxc-c500` 标签明确针对 C500 |

## 第一批可证伪问题

以下是实验问题，均不是既成硬件结论。

1. **身份与线程组：**C550 设备查询、原生编译器目标、Triton target 与实际 block 线程数能否对应？记录原生目标和兼容 capability 为不同字段，验证 32、64、128 线程边界以及尾部正确性。
2. **计时范围：**同一 kernel 的单次事件、批量事件均摊和主机同步计时如何变化？先用空 kernel 与可控工作量核对量级，再分别报告启动开销、批量平均和测量噪声。
3. **访存形状：**顺序、跨步、错位和向量化访存是否产生稳定差异？保持逻辑读取字节数与正确性相同，记录实际指令和缓存状态，不把有效带宽自动称作 HBM 带宽。
4. **容量与复用：**工作集扫描是否出现可重复的延迟或带宽转折？热复用、独立地址和明确重置各做一组；单个转折不足以命名缓存层级或容量。
5. **WSM 访问：**共享内存步长、广播与 padding 如何影响延迟？先测冲突模式，再解释 bank 结构；不预设 NVIDIA 的 bank 数和 bank 宽。
6. **编译决策：**在同一已验证计算上，`num_warps`、`num_stages`、`basic/cpasync` 是否改变产物、正确性和时间？只有本机版本支持的选项才进入实验，不由选项名称推定硬件指令。

[runtime]: https://developer.metax-tech.com/api/client/document/preview/567/C500_RuntimeAPIProgrammingGuide_CN.html
[release-overview]: https://developer.metax-tech.com/api/client/document/preview/发布说明/MXMACA_发布说明/曦云C500系列/latest/split_files/概述.html
[release-changes]: https://developer.metax-tech.com/api/client/document/preview/发布说明/MXMACA_发布说明/曦云C500系列/latest/split_files/新增特性及变更.html
[release-limits]: https://developer.metax-tech.com/api/client/document/preview/发布说明/MXMACA_发布说明/曦云C500系列/latest/split_files/已知问题和使用限制.html
[guide]: https://gitee.com/metax-maca/mxmaca-performance-tuning-guide/blob/main/README.md
[vector-guide]: https://gitee.com/metax-maca/mxmaca-performance-tuning-guide/blob/main/guide/ch1.初探异构编程.vectoradd.md
[profiler]: https://developer.metax-tech.com/api/client/document/file/211/preview/?file_type=pdf
[mxvs]: https://developer.metax-tech.com/api/client/document/preview/996/index.html
[triton-readme]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/README.md
[triton-driver]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.py
[triton-driver-c]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/driver.c
[triton-compiler]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/backend/compiler.py
[triton-codegen]: https://github.com/MetaX-MACA/mcTriton/blob/7dd407c26568fceaca44cb894138e5202d369805/third_party/metax/triton_metax.cc
[kernel-timer]: https://github.com/MetaX-MACA/vLLM-metax/blob/f2fcc59c314f1fbc7897f46d465e5f7c35900e8a/csrc/libtorch_stable/quantization/awq/hgemv_selector.hpp
[mctvm]: https://github.com/MetaX-MACA/mcTVM
