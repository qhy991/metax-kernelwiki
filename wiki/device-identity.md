# C550：先确认正在讨论哪一种身份

一个产品可以同时出现多种架构名称。它们回答不同的问题，不能互换：

| 字段 | 含义 | 核实方式 |
| --- | --- | --- |
| 产品名 | 实际板卡，例如 MetaX C550 | `mx-smi` 和 native runtime |
| 设备 ISA | 设备报告的指令目标 | `macainfo` |
| native codegen family | 当前编译器实际选择的目标族 | 编译参数、生成物、编译器工具 |
| MACA capability | MACA runtime 的 major/minor | 编译本机 SDK 属性查询 |
| Triton architecture | 后端 API 用来选择编译路线的值 | 安装版本的 driver/compiler 源码与实际 metadata |

已有 [open-cake-ir 调查](https://github.com/qhy991/open-cake-ir/blob/main/docs/metax-c550-bringup.md) 曾在 C550 上分别观察到 `XCORE1002`、`xcore1000`、native capability `10.2`。这些是历史外部观察；本库会保留自己的采集日期、SDK 与设备，而不把历史值当作新一次实测。

## wave 宽度影响什么

连续线程访问连续元素，才有条件形成合并访存；跨 wave 的数据交换不能沿用假设 32 lane 的 shuffle 或 ballot 写法。64 lane 是官方系列文档及当前 MACA Triton 后端的声明，设备测量仍单独记录。

当从 CUDA kernel 移植时，检查 `32`、`0xffffffff`、`lane & 31`、`threadIdx.x >> 5` 等是否承担硬件语义。算法 tile 常量可以是 32，但不能因此把硬件执行组也写成 32。

**待验证问题**：非整 wave 的 block 是否正确工作？64/128/256/512 的 block 配置在相同有效访问量下是否改变延迟？这些问题需要 kernel 输出和计时，设备属性查询不能回答。

## 头文件别名不等于两个独立观察

MACA 3.5.3 的头文件检查发现 `mc_runtime_api.h` 有 `#define waveSize warpSize`，而类型定义中包含两个名称。源码打印两次名称可能读取同一字段。属性查询应以 SDK 枚举编译并逐项检查返回码；记录它们来自同一运行时，不能把两个相等结果称为相互独立的硬件验证。
