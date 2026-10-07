# 研究进展

## 2026-10-07：建立 C550 证据基线

目标是回答：怎样从 MACA 的原生接口、编译器和小型反例实验中提取对 C550 内核开发有用的知识？

已读取两个参考库；官方 MACA 文档与 mcTriton 源码索引正在整理。C550-1 的 SSH、MACA SDK 与原有测试容器可见。观测到八张 GPU 均持有 vLLM worker 上下文，每卡占用约 64 GiB；利用率为零不代表可以独占。未干预这些工作。

首轮实验问题：

1. 产品、ISA、原生 runtime 属性和编译目标是否一致地标识所选 C550？
2. 带尾部的 copy 在不同 block 大小下是否逐元素正确？
3. 固定输出元素数时，stride 1/2/4/8/16 的读取如何影响事件区间和有效数据率？
4. 空 kernel 的批次事件时间随批量大小如何变化？能否区分固定事件开销与每次 dispatch 的间隔？

后续按证据推进：复制/转置 → shared memory padding 与 barrier → reduction/shuffle 的 64-lane 边界 → 矩阵编译路线与 profiler。每轮只提出数据能区分的机制解释；没有 profiler/汇编依据时不推断 bank 数、cache line 或矩阵指令吞吐。
