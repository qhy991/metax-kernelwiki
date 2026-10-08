# C550原生WMMA：严格精确验收失败与可复现残差

在本次C550 / MACA 3.5.3.18 / MXCC `1.0.0 (6477545d4d)` 环境，原生16×16×16 FP16 WMMA及float累加fragment能够编译和执行，但**没有满足这组输入的严格精确数值合同**。首次12case中，K=0和1×1×1精确通过，其余10case共177个输出不等于预先固定的参考值。输入与guards完整，输出全部有限。

首个失配位置是16×16×16的C[0,0]：参考为`-0.5`（`0xbf000000`），设备输出为`-0.5000000596046448`（`0xbf000001`）。同一冻结二进制在两个独立进程及一份诊断trace中，完整输出位均重现首次结果。[全部原始输入、输出与诊断记录](../data/results/20261008-wmma-exact-diagnostic.json)明确保存`correctness.passed=false`及`performance_accepted=false`；没有放宽容差，也不从这些计时报告性能。

## 预先固定的矩阵与数值合同

源码`ef22b51`采用native `-x maca -offload-arch=xcore1000`，包含`mcr/mc_runtime.h`与当前编译器资源头`__clang_maca_mma_functions.h`，调用`mxmaca::wmma`。这是本机版本的实际入口；官方示例的`mma.h`位于cu-bridge包含路径，其宏条件不能直接当作native入口已验证。

一个完整64-thread block处理一个16×16输出tile。A按row-major、B按column-major分成16宽的K块；每个operand固定存四块，不足M/N/K的位置在主机准备阶段补+0。每个线程都经过相同的`ceil(K/16)`循环，float累加fragment初始化为0，调用安装头文件中的四参数`mma_sync`，最后以row-major存出全部256个值。K=0也执行fill和store，不提前退出。packing与传输不在事件计时内。

逻辑输入为：

```text
a_num(i,k) = ((67*i + 13*k) % 31) - 15
b_num(k,j) = ((17*k + 5*j + 3) % 29) - 14
A = a_num / 16; B = b_num / 16
C_ref(i,j) = sum_k a_num(i,k)*b_num(k,j) / 256
```

这些输入能由binary16精确表示，单个乘积为整数/256。K≤64时，任意子集或部分和的分子绝对值不超过`64×15×14=13440`，小于2²⁴。因此一串正确舍入的FP32乘法和加法能精确表示这些中间量；改变这种求和的结合顺序本身不需要产生残差。

CPU oracle独立使用逻辑i/j/k整数点积，不调用WMMA、不使用fragment的lane映射。验收要求所有256个输出有限且数值精确相等，+0/−0等价；两侧各64个guard按位检查。另有错误B布局、漏K块、未写padding、非有限值和FP16舍入后的错误输出等CPU负对照。[实现与复现命令](../experiments/wmma/README.md)保留完整合同。

## 首次结果与独立复现

下面是对首次保留输出的全量诊断。每case检查完整16×16区域，包含逻辑M/N外的零输出：

| M×N×K | 精确不等的输出数 | 最大绝对残差 |
| --- | ---: | ---: |
| 16×16×0 | 0 | 0 |
| 1×1×1 | 0 | 0 |
| 16×16×16 | 30 | 5.96046448e-08 |
| 15×16×16 | 28 | 5.96046448e-08 |
| 16×15×16 | 28 | 5.96046448e-08 |
| 15×15×15 | 25 | 5.96046448e-08 |
| 7×9×17 | 3 | 5.96046448e-08 |
| 15×16×31 | 19 | 1.1920929e-07 |
| 16×15×32 | 18 | 1.1920929e-07 |
| 16×16×33 | 15 | 1.1920929e-07 |
| 9×7×63 | 1 | 4.76837158e-07 |
| 16×16×64 | 10 | 4.76837158e-07 |

原严格CLI在第三个case的首个失配处停止，所以原计划中的逆序和性能trace没有启动。之后另建**失败复现诊断**：保持原`ef22b51`二进制、输入、oracle、预热与计时设置不变，单独执行两次dense K16、一次dense K64，再采一份K16诊断trace。它们的A/B文件及全部输出words（含guards）均与首次对应case一致；严格checker仍返回失败。原失败没有被重跑结果覆盖。

独立诊断重新检查了首次CLI尚未检查到的后续输出，并对全部16个实际case保留了32,768个输入halfwords、4,096个C值和2,048个guards。所有输入符合预先声明的packed布局，所有C值有限、所有guards完好；参考为零和逻辑padding的位置也保持精确零。把实际输入逐项乘法、每次累加都舍入到FP32的CPU重放，同样得到精确参考值。

这些检查支持排除host打包错误及通常的FP32求和重排作为当前解释，不能据此定位硬件缺陷。尚未保存device侧A/B回读，也没有在GPU上运行独立标量FP32对照，内部算术、编译器、输入/输出路径及硬件的责任仍未分离。

## 误差指标不能省略定义

最大绝对残差为`4.76837158203125e-7`。最大相邻FP32距离出现在M16/N15/K32的C[0,9]：

| | 参考 | 实际 |
| --- | --- | --- |
| 数值 | 0.00390625 | 0.0039062313735485077 |
| FP32 bits | `0x3b800000` | `0x3b7fffb0` |

这里按单调FP32编码计算两数之间的**相邻可表示值步数**，把两个符号的零视为同一个值，得到80步。参考恰好在2的幂边界；若改用参考值向上的spacing作分母，会得到40，所以不能不加定义地称为“80 ULP”。这只是所测数据的观察上界，不是通用容差建议。

## 文档与profiler的证据边界

[官方C++指南3.5.3.x的WMMA章节][wmma]描述D=A×B+C、fragment类型和全warp参与要求，[类型表][types]列出FP16输入与float累加fragment。但本次核查没有找到该章节对中间精度、舍入方式、误差界或逐步IEEE FP32乘加等价的保证。同页half/half2算术或转换函数的舍入条款不能转用于WMMA。因此当前结论是**强exact合同失败、数值机制未解释**，不是已证实的厂商缺陷。

诊断trace含110个`wmma_tile_kernel`事件，报告28regs、static/dynamic shared和每线程private均为0，重编译标志110个false且无缺失。这不能识别造成残差的算术机制。原始事件批次和trace时长仅作追溯保留；trace时间单位仍未独立验证，不发布吞吐、加速比或每条MMA延迟。五个device worker及一个profiled应用已退出并完成释放检查。

wiki把本页标为`locally-measured / device-correctness`，表明它是直接测得的数值诊断；原性能入口`local-measurement`仍必须通过完整正确性检查。索引允许保存失败事实，不会把失败变成通过。

下一轮将在后继源码与新诊断计划中保存device输入回读，并加入读取同一packed输入的标量FP32 GPU对照；保留当前失败，不在旧运行中改容差或替换实现。本轮没有向open-cake-ir的Compiler、Target或校准提升。

[wmma]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#warp-matrix
[types]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#nhvxy67mk8uv1
