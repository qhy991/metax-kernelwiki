# C550 wave64：shuffle子组、整数归约与逻辑尾部

在一张C550、MACA 3.5.3.18与MXCC `1.0.0 (6477545d4d)` 上，完整64/128-thread block的直接shuffle及signed-int归约通过了本轮边界检查。源码固定为`d01ef59`；正序20case、逆序20case和两个单case trace累计检查39,744个输出值及2,688个guard words，全部正确。[完整结果](../data/results/20261008-wave-collectives.json)包含实际输出值、420个事件批次和全部检查摘要。

这里区分三个量：物理wave有64个lane；shuffle的width可以将其划成更小子组；逻辑数据长度n可以小于实际参与线程数。改变后两个量，不表示物理wave宽度变了。

## 一份输入怎样区分这些边界

输入是128个int32值`1..128`。所有物理线程都执行同样的collective，使用显式的64-bit `unsigned long` mask `0xffffffffffffffffUL`。每线程保存九个结果：width64的source 0/31/32/63、width32的同四个source，以及一次`__reduce_add_sync`结果。编译断言检查mask与int的宽度；执行时要求设备名为C550、只有一个可见设备且runtime报告wave64。

下表来自完整128-thread、n=128的实际结果。同一列所列线程段内，每个线程都得到相同的相应结果：

| 操作 | thread 0–31 | thread 32–63 | thread 64–95 | thread 96–127 |
| --- | ---: | ---: | ---: | ---: |
| shuffle width64, source0 | 1 | 1 | 65 | 65 |
| shuffle width64, source31 | 32 | 32 | 96 | 96 |
| shuffle width64, source32 | 33 | 33 | 97 | 97 |
| shuffle width64, source63 | 64 | 64 | 128 | 128 |
| shuffle width32, source0或32 | 1 | 33 | 65 | 97 |
| shuffle width32, source31或63 | 32 | 64 | 96 | 128 |
| full-mask整数求和 | 2080 | 2080 | 6176 | 6176 |

[官方3.5.3.x shuffle契约][shuffle]规定直接索引在width子组内按`source % width`选择。因此width32、source32选择的是调用者所在子组的第0个元素，并非整个wave的lane32。上半子组仍有自己的源；第二个物理wave也不读取第一个wave的输入。

整数求和按各自完整wave进行。把它误写成32-lane组会得到528、1552、2576、3600；把整个block混成一次求和会得到8256。CPU负对照确认这些结果都会被拒绝。本轮实际每个参与lane都得到其所属wave的2080或6176，而不只是检查lane0。

## 逻辑尾部补零，参与线程不减少

thread i在`i<n`时读取输入，否则提供0；它仍执行所有collective并写九个结果。n=0也保留完整物理线程，不能把“不存在有效输入”实现成线程提前退出。

例如128-thread、n=65时，第二个wave只有thread64提供值65，其余输入都补零。thread127的输入虽然为0，其归约结果仍为65；width64/source0也得到65，而其所在上半width32子组的广播全为0。这些输出都在实际检查范围内。只检查前n个线程，会漏掉这类差别。

| 物理block | 逻辑n |
| --- | --- |
| 64 | 0、1、31、32、33、63、64 |
| 128 | 上述七种，再加65、95、96、97、127、128 |

CPU oracle用独立的输入list切片、子组索引和整数sum生成期望，不复用GPU的lane位运算或归约实现。每个物理线程的全部九个int32值和两侧各32个guard均按位比较；未写的`0xffffffff`与包括0在内的任何期望值都不同。逆序复验是在新进程中检查相同语义，不是性能处理或独立随机输入试验。

## 接口、编译和trace各证明什么

[官方整数归约契约][reduce]描述mask参与条件与返回语义。本机SDK的`mxgpu_llvm/lib/clang/19/include/__clang_maca_device_functions.h`包含整数64-bit mask入口，并委托给一个包含排列与累加循环的helper；本轮原生`-x maca -offload-arch=xcore1000`编译和设备输出进一步验证了上述有限用例。这个接口名不保证最终生成单条硬件指令，头文件实现也不能代替实际代码生成证据。

两个独立trace分别对应block64/n64与block128/n128，各有110个同名kernel事件，block.x分别报告64/128；均报告16个寄存器、static/dynamic shared为0、每线程private为0。函数查询也报告16个寄存器、shared/local为0。每份trace的110个重编译标志均为false，没有缺失；四个运行的新binary-cache目录均保持为空。这些是API或工具报告，不推导物理驻留、指令数或硬件峰值。

保留的计时属于完整九通道kernel的事件批次均摊，可能包含host供给间隙；不能除以九后称为某个intrinsic的延迟。trace导出时间单位仍未独立核验。当前没有由此建立吞吐排名、算法加速比或框架级收益。

## 复现与尚未覆盖的条件

[探针及命令](../experiments/wave_collectives/README.md)保留输入、mask、输出布局、计时和拒绝规则。结果JSON中的`output_words_uint32`直接来自设备保留的.i32文件，包含guards；可结合一次保存的`input_words_uint32`复核，不能把oracle生成值当成设备输出。全部四个device worker及两个profiled应用均已退出并完成释放观测，分配范围仍是合作式`local_serialized`。

本轮仅覆盖位置编码的非负int32输入、0填充、完整物理wave与完整64-bit mask；不证明稀疏mask、inactive源lane、部分物理wave、负数/溢出、浮点归约、down/xor/vote或内存顺序。寄存器shuffle/归约不能代替[文档单独定义的同步与内存顺序][sync]。

下一步先查清mask参数类型与兼容入口的关系，再冻结新的对照；不会修改本轮结果或把另一入口的语义混进此合同。本轮没有向open-cake-ir的Compiler、Target或校准提升。

[shuffle]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#warp-shuffle
[reduce]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#warp-reduce
[sync]: https://developer.metax-tech.com/api/client/document/preview/编程参考/MXMACA%20C%2B%2B编程指南/曦云C500系列/3.5.3.x/split_files/c_语言扩展.html#pddnkif8w7ir1
