# 512 是哪一种上限：C550 的 launch bounds 与运行时重编译

在本次 C550 / MACA 3.5.3.18 实验中，函数属性报告的 `maxThreadsPerBlock=512` **没有阻止 1024-thread block 正确执行**。默认 copy 走到了带重编译标志的运行时变体；对同一函数显式设置 `__launch_bounds__(1024)` 后，函数属性变为 1024，独立 trace 中未再观察到这条重编译路径。

这是一份具体 kernel、MXCC 版本和运行条件内的结论。不能把 CUDA 的完整 launch-bounds 语义、某个 trace 字段或本例的启动时间直接推广到其他 kernel。

## 三个字段分别属于谁

| 观察入口 | 本次值 | 可以回答的问题 |
| --- | ---: | --- |
| `mcDeviceProp_t.maxThreadsPerBlock` | 1024 | 该设备的 runtime 接口声明的 block 上限 |
| 默认 copy 的 `mcFuncGetAttributes` | 512，即使已经启动过 1024-thread block | 该函数接口报告的默认编译配置；本例中不是动态执行的硬件上限 |
| 默认 copy 在 1024-thread trace 中的 `max_block_size` | 1024，949 条报告、72 条缺失 | 这次执行变体的工具资源字段；缺失项保持缺失 |
| 显式 `__launch_bounds__(1024)` 的 copy 函数属性 | 1024 | 本机编译器接受并反映了此单参数声明 |

安装的 SDK 头文件 `mcr/mc_runtime_api.h` 在 `mcFuncGetAttributes` 的说明中区分了未指定 launch bounds 时的默认 512 与硬件支持的 1024。[官方 3.5.3.x 运行时指南的 binary-cache 示例][runtime-cache]也说明，示例中的 1024-thread launch 会触发运行时重编译。头文件和文档提出可检验预期；以下设备实验检验了该预期在本 copy 上的表现。

## 默认函数的边界实验

源码 `088783d` 的 [18-case 边界实验](../data/results/20261007-block1024-boundary.json) 覆盖 512/1024 两种 block、单元素、小于/等于/大于 block 的长度及约 16 MiB 的 copy，还包含两个 empty kernel case。16 个有输出 case 的 **8,405,022 个 payload 元素及 1,024 个 guard words 全部正确**。

这一轮每个 case 先记录一次 host launch 到完成同步的时间，再执行原来的 20 次预热和 10×100 次计时。大 copy 在 512/1024 配置下的事件均摊中位数为 45.403/40.180 µs；这是单次固定顺序扫描，不能据此选择通用最优 block。首次遇到 1024-thread copy 时，host 完成区间约 162 ms。该区间包含当时触发的加载、编译、提交及同步，不能独立解释为 JIT 编译耗时。

后继分别在新进程、空的 run-local binary-cache 目录中采集单 case trace：

| 默认 copy 的 block | runtime 函数属性 | trace 的 `max_block_size` | `is_recompiled` 已报告值 | 新 cache 文件 |
| ---: | ---: | ---: | --- | ---: |
| 512 | 512 | 512 | 949 条 false，72 条缺失 | 0 |
| 1024 | 512 | 1024 | 949 条 true，72 条缺失 | 2 |

每份 trace 均含 1021 个真实 GPU kernel event：1 次首发诊断、20 次预热和 1000 次被计时的 launch。**949 条 true 不是发生了 949 次编译**；它是执行事件描述符携带的标志。两个 cache 文件也不是编译次数。

## 只改变一个声明的对照

源码 `bc9f748` 提供同一 copy 函数体的两个编译配置。默认宏值 0 不加 annotation；宏值 1024 只给 copy 添加：

```cpp
__global__ __launch_bounds__(1024)
void copy_kernel(const float* input, float* output, uint64_t n);
```

这是原函数定义的声明示意。完整实现见 [probe.cpp](../experiments/native/probe.cpp)，使用方法见 [native README](../experiments/native/README.md)。两侧使用相同 commit、编译器、目标、block=1024、输入与 oracle，各自在新进程和空的二进制缓存目录内执行。

| 编译配置 | runtime 函数属性 | trace `max_block_size` | 重编译标志已报告值 | 新 cache 文件 | 首次 host 完成区间 |
| --- | ---: | ---: | --- | ---: | ---: |
| 默认声明 | 512 | 1024 | 949 true，72 缺失 | 2 | 164.734 ms |
| 显式 bound=1024 | 1024 | 1024 | 949 false，72 缺失 | 0 | 2.659 ms |

两份独立 trace 的最终输出均逐位正确。显式声明版本另外通过了完整 18-case 边界检查；未改动的 empty kernel 属性仍是 512。[原始样本、属性和覆盖率](../data/results/20261007-launch-bound-control.json)保留了这些区别。

这个对照支持：**对该 copy，预先声明 1024 的函数 bound 可以避免所观察到的运行时重编译路径。** 这里只各有一次新进程 trace，表中的首次完成时间不是重复测量得到的加速比，也不是纯 kernel latency。未观察到新文件或标志，不等于证明任何位置都没有发生编译工作。

## 什么时候值得尝试，什么时候还需验证

如果已知某个函数需要以超过默认编译约束的 block 执行，可以把显式 bound 作为减少首次启动工作的候选。先检查对应 SDK 的实际接口，再验证全量输出、函数资源、运行时变体及稳态性能。编译器可能因声明改变寄存器分配和其他代码生成决策；本例不能替其他函数作决定，也不支持一律改用 1024-thread block。

本轮没有测试第二个 `__launch_bounds__` 参数、超过显式 bound 的 launch、其他 SDK/GPU 或大算子的端到端收益。二进制 shader cache 与 L2/DRAM 数据缓存是两件事；这里没有控制后两者，也没有内存流量计数器。trace 导出的 `ts/dur` 单位仍未独立核实；上述 host 时间来自 C++ 单调时钟，事件时间来自 `mcEventElapsedTime`，未将原始 trace 数值转换成微秒。

[runtime-cache]: https://developer.metax-tech.com/api/client/document/preview/编程参考/运行时API编程指南/曦云C500系列/3.5.3.x/split_files/编译和调试.html#binary-cache
