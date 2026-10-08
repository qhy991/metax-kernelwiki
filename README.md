# MetaX KernelWiki

面向 **MetaX C550** 的内核知识库：从官方资料提出可检验的问题，在精确记录的 MACA 环境中运行小实验，再把有边界的结果写回机制页。

借鉴 [metal-kernelwiki](https://github.com/qhy991/metal-kernelwiki) 的轻量检索和 [bw1100-kernelwiki](https://github.com/qhy991/bw1100-kernelwiki) 的机制与证据写法。独立于 open-cake-ir；这里的研究结果不自动修改 Compiler Target、校准或资格集合。

## 阅读入口

| 目的 | 入口 |
| --- | --- |
| 看现阶段做了什么、下一步测什么 | [研究进展](docs/research-log.md) |
| 区分产品、原生 ISA 与兼容架构值 | [设备身份](wiki/device-identity.md) |
| 查看wave64、shuffle子组与归约尾部 | [collective边界实测](wiki/wave-collectives.md) |
| 查看同结果转置的优化与尾部验证 | [转置分块与 padding](wiki/transpose.md) |
| 查看固定地址集合的访存复验 | [读取排列与时间](wiki/memory-order.md) |
| 理解 512/1024 线程与首次启动 | [launch bounds 与运行时重编译](wiki/launch-bounds.md) |
| 查编译器与 Triton 原始依据 | [工具链](docs/toolchain.md) · [资料索引](docs/sources.md) |
| 判断一个性能数字说明了什么 | [实验方法](docs/methodology.md) |
| 重现实验 | [原生探针](experiments/native/README.md) · [读取排列](experiments/memory_order/README.md) · [转置控制](experiments/transpose/README.md) · [wave collectives](experiments/wave_collectives/README.md) |
| 查参考库和来源边界 | [来源沿革](docs/provenance.md) |

## 当前实测范围

首轮已在 C550 / MACA 3.5.3.18 完成 **49 个 case**，逐位检查 **22,042,413 个有效输出**及边界 guard。公开记录保存全部 **490 个计时批次**；四进程平衡顺序确认和含 1020 个 GPU kernel 的独立 trace 也已完成。缓存策略、时钟、导出时间单位及并发覆盖限制见[研究进展](docs/research-log.md)。

后继实验覆盖 [1024-thread 与重编译路径](wiki/launch-bounds.md)、[固定地址集合的读取排列](wiki/memory-order.md)，以及[同结果转置的分块、padding 和固定容量行距对照](wiki/transpose.md)。另有[完整wave的shuffle与整数归约](wiki/wave-collectives.md)边界验证，并公开实际整数输出。各轮保留独立源码、完整输出检查、进程级复验和 profiler 记录；具体条件与未覆盖范围以对应机制页为准。

## 检索

Python 标准库即可，无需 GPU 或网络：

```sh
python3 scripts/wiki.py list
python3 scripts/wiki.py search wave
python3 scripts/wiki.py show maca-event-timing
python3 scripts/wiki.py validate
```

`data/catalog.json` 是唯一检索目录；正文和原始来源通过路径/URL 引用。记录分别标注 `confidence`、`evidence_scope`、具体环境以及限制，不用一个“verified”覆盖文档、编译与设备测量的区别。

## 证据规则

- 文档声明、上游源码、历史观察、此次本机结果各自说明来源。C500 系列资料不自动成为 C550 的测量值。
- 先准备输入和 CPU oracle，再通过机器现有分配入口获取设备；保存输出并退出设备进程后才做主机分析。原始运行输出在仓库外按 run id 保留。
- 只有完整输出正确才报告该 case 的性能。事件批次时间包含事件之间的整个设备执行区间；记录应用的状态处理和运行时缓存策略是否经过验证，外部活动未排除时明确写出。
- 原始 JSONL、编译日志、失败与负对照保留；整理后的有限结果进入 `data/results/`。不覆盖旧实验来“修正”结论。
- 一个提交对应一版实验源码。后继问题用新的 run id；不会把 C550 的观察提升为其他 MetaX 产品的事实。

GPU 操作遵循 [gpu-infra 的租约生命周期](https://github.com/qhy991/gpu-infra/blob/main/skills/gpu-infra/SKILL.md#gpu-lease-lifecycle)。本库不实现第二个分配器。
