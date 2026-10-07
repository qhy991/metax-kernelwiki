# MACA trace：成功采集与成功退出是两件事

本机使用 `mcTracer 3.5.3.18-ef9e10e` 对首轮 copy 的 block 256 配置做了独立采集。采集运行的事件计时与无采集实验分开，不能用带 tracer 的数值替换原来的性能样本。

## 输出目录失败仍可能返回 0

`20261007-trace-01` 给 `--odname` 传入绝对路径。该版本把它拼到当前工作目录后，导致目录创建失败；日志包含 `FATAL`，但退出码为 0。应用仍生成了输出，因此仅检查应用输出或进程退出码都不能证明 profiler 成功。

后继 `20261007-trace-02` 在新的运行目录中使用相对的 `--odname trace`，生成可解析 JSON，并通过这次 copy 的完整 CPU 输出检查。第一份失败记录保留，没有补写成成功。

## 本次 trace 实际覆盖什么

共 5171 个 trace event，其中 **1020 个 GPU copy kernel event** 对应 20 次预热和 1000 次被计时的 launch。它们与 host `mcLaunchKernel` 的 correlation id 一一对应；不能只把 host API 调用数当作 GPU dispatch 数。

这 1020 个 kernel event 均报告 block 256、grid 16385、每线程 6 个寄存器、静态/动态 shared memory 为 0、private memory 为 0。这里保留的是该工具对这份产物的报告，不是通用寄存器上限或 occupancy 标定。`max_block_size=512` 仅在部分事件中出现；缺失值不能补成 0。它与设备 runtime 报告的最大 1024 线程是不同层级的字段。

trace 元数据中的名称 `C500` 是工具标签；本轮 native runtime 的精确设备名仍是 `MetaX C550`，不会因兼容标签而更换目标身份。

## 时间单位尚未单独验证

JSON 没有声明时间单位。安装的 MCPTI 头文件将 activity timestamp 注释为 ns，而导出的 timestamp 数量级和事件批次跨度与 ns 解释一致；这仍不足以单独验证 exporter 没有做变换。本库保存 `dur` 原始数值，标记单位未验证，不直接把它作为已校准的微秒 kernel latency。

去掉 20 个预热后，1000 个 GPU event 的原始 `dur` 为 median 48640，范围 [46848, 52224]。这些数值来自带 tracer 的运行。完整分析见[采集结果](../data/results/20261007-trace-02.json)。

后续需要对 exporter 的单位契约或实现作独立确认，再比较相同采集运行中的 kernel 区间和 event 区间。trace 没有提供已验证的 cache/DRAM 性能计数器，本次不会据此归因访存瓶颈。
