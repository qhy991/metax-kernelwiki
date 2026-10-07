# 研究进展

## 2026-10-07：建立 C550 证据基线

目标是回答：怎样从 MACA 的原生接口、编译器和小型反例实验中提取对 C550 内核开发有用的知识？

已读取两个参考库；已整理官方 MACA 文档与 mcTriton 固定源码索引。C550-1 的 SSH、MACA SDK 与原有测试容器可见。观测到八张 GPU 均持有 vLLM worker 上下文，每卡占用约 64 GiB；利用率为零不代表可以独占。未干预这些工作。

首轮实验问题：

1. 产品、ISA、原生 runtime 属性和编译目标是否一致地标识所选 C550？
2. 带尾部的 copy 在不同 block 大小下是否逐元素正确？
3. 固定输出元素数时，stride 1/2/4/8/16 的读取如何影响事件区间和有效数据率？
4. 空 kernel 的批次事件时间随批量大小如何变化？能否区分固定事件开销与每次 dispatch 的间隔？

后续按证据推进：复制/转置 → shared memory padding 与 barrier → reduction/shuffle 的 64-lane 边界 → 矩阵编译路线与 profiler。每轮只提出数据能区分的机制解释；没有 profiler/汇编依据时不推断 bank 数、cache line 或矩阵指令吞吐。

## 首轮设备结果

`20261007-native-01` 使用源码 `cbdea92`，在一张 C550、MACA 3.5.3.18 上完成 49 个 case（45 个有输出、4 个空 kernel），检查 22,042,413 个有效输出与 2,880 个 guard words，全部通过。16 个 CPU 负对照/协议测试在固定源码 checkout 通过。GPU 进程退出后才执行完整 CPU 检查；释放观测未发现遗留 MACA 锁或 GPU 进程。

[完整数值与 490 个原始事件批次](../data/results/20261007-native-01.json) · [机制解释](../wiki/memory-access.md)。

约 16 MiB 的逐元素 copy 在本次固定顺序扫描中，block 64/128/256/512 分别为约 67.7/57.6/47.5/43.4 μs 的事件批内均值。stride sweep同时改变事务访问形状和工作集跨度，不能从趋势单独归因 cache 或合并访存。[四个独立进程的平衡顺序确认](../data/results/20261007-copy-confirm.json) 全部通过，四轮排序一致；进程内 `T64/T512` 为 1.528 [1.522, 1.532]。分析单位是进程，不把 40 个批次称为 40 次独立复现。

## Profiler 的一次失败与后继采集

`20261007-trace-01` 发现 mcTracer 3.5.3.18 将绝对 `--odname` 拼到工作目录后，报路径创建失败，但工具退出码仍为 0。失败记录保留。`20261007-trace-02` 在新目录使用相对 `--odname trace`，生成 trace 并通过该次 copy 的完整输出检查。[trace 内容验收](../wiki/profiling.md)确认 1020 个真实 GPU kernel 事件与 host launch 一一对应；导出时间单位仍标为未独立验证，保留原始数值，不转换为已校准的微秒。不能只检查文件存在或工具退出码。

## 下一轮问题

- 设备允许每 block 1024 线程，而这份 kernel 的部分 trace 资源字段报告 `max_block_size=512`。后继源码要检查 512/1024 的编译与运行时行为、完整正确性及是否出现 runtime recompilation，不能把两种上限混为一谈。
- gather 的跨度与线程地址间距同时改变；设计相同跨度的对照，再检查跨进程稳定性。
- 核实 mcTracer exporter 的时间单位契约；之后才能把同一次 trace 的 kernel 区间与 event 区间作带单位比较。
- 这些是本库知识与探针结果；本轮没有向 open-cake-ir 的 Compiler、Target 或校准提升。
