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

## 2026-10-08：512/1024 函数约束与重编译路径

设备执行和采集于 2026-10-07 完成，次日汇总。后继源码 `088783d` 的 18-case 边界检查通过：1024-thread block 可以正确执行，但默认函数的 `mcFuncGetAttributes.maxThreadsPerBlock` 仍返回 512。单 case trace 则将默认 1024-thread copy 的执行变体报告为 `max_block_size=1024`，在 949 个有此字段的事件中 `is_recompiled=true`，另 72 个事件缺失该字段。

再用同一提交 `bc9f748` 编译默认与显式 `__launch_bounds__(1024)` 两个版本，保持 copy body、block=1024、输入、oracle 和其余编译参数不变。显式版本通过完整 18-case 边界检查；独立 trace 中函数属性变为 1024，949 个有值事件的重编译标志均为 false，新的 binary-cache 目录没有产物。默认对照仍报告 512/重编译标志 true，并产生两个 cache 文件。两个首次 host 完成区间为 164.734/2.659 ms，只有各一次受 profiler 影响的观测，不发布加速倍数。

[机制页](../wiki/launch-bounds.md) · [默认边界和 trace 原始样本](../data/results/20261007-block1024-boundary.json) · [同源显式声明对照](../data/results/20261007-launch-bound-control.json)。

资源方面，第一次旧用户级锁请求被其他卡上的合作任务拒绝，未执行 probe；该失败保留。后继调用器显式使用节点已部署的设备级 owner，以同一套锁协议获取设备 0，没有停止其他任务、删除锁或增加分配器。上述设备进程全部退出并完成释放检查。

文档方面，重新核实了官方 3.5.3.x 运行时指南与 mcTracer 手册；初轮 `/preview/567` 的活动版本实际为 3.0.0.x，已保留追溯备注并修正当前入口。mcTracer 导出时间单位仍未找到明确契约，继续保留原始值。

本轮结论进入本 wiki，没有修改或提升到 open-cake-ir 的 Compiler、Target 或校准。之前 `eb6021e` 的 GitHub CPU workflow 已确认成功；新增控制源码 `bc9f748` 在干净独立 checkout 上通过 39 项 CPU 检查。

## 2026-10-08：固定地址集合的读取排列

源码 `0e093ce` 的新探针对每个 N 都读取完整 `0..N-1` 集合，连续写入输出，在进程内复用同一 input/output allocation；只改变读地址排列。CPU 以独立矩阵转置索引验证结果。18 个参数 case 通过全量检查，之后对 N=2^24 用预先固定的六种平衡顺序复验，36 个 case 全部通过。

固定整体跨度后仍有明显非单调现象：进程内 `T(s=6)/T(s=12)` 为 3.284 [3.258, 3.288]，六轮均同向。不能将更大的相邻读间隔直接解释为更慢，也不能仅以整体输入跨度解释本次差异。三个独立单 case trace 各有 110 个 kernel，均报告8个寄存器、shared/private为0，未观察到重编译标志变化；尚无内存流量计数器来指定唯一原因。

[机制页](../wiki/memory-order.md) · [570 个事件批次、全部六进程和三个 trace 的公开记录](../data/results/20261008-memory-order.json)。一套输入在多个参数/进程中被重复检查，输出元素检查次数不能说成不同随机输入数量。全部十个设备进程已退出，并通过分配释放检查。

本轮没有向 open-cake-ir 提升；`9411468` 的公有仓库 CI 已通过，新探针固定提交在独立 checkout 上通过52项CPU检查。

## 2026-10-08：同结果转置的共享内存分块

源码 `bf1db0f` 增加同一 row-major FP32 transpose 的 direct、tile64、tile64_pad1 三种实现。19个shape共57case全部通过，包含1维细长、31×33/33×31、63/64/65边界以及两个大ragged矩阵。有效store与其对应shared loader的guard一致，整个block无条件经过barrier。

三个大尺寸的六进程54case确认也全部通过。未padding的tile64相对本次direct的进程内中位加速比分别为9.183、7.523、2.833；padding则没有统一方向：262144×64近似中性、65536×256略慢、4096×4096略快。不同操作数、block数、每线程工作与shared/barrier共同改变，因此收益不能只归因于一个机制。

[机制页](../wiki/transpose.md) · [1140个原始事件批次与三份独立trace](../data/results/20261008-transpose.json)。三份trace各110个kernel，报告direct/tiled寄存器为14/13，shared为0/16384/16640字节；未采bank-conflict或DRAM计数器。官方C500资料只作为先验，没有转换为C550 bank常量。

全部十个设备进程及被profile的应用均已退出，未保留本任务分配。此轮仍为独立native实验，没有向open-cake-ir的Compiler/Target/校准提升。固定源码在独立checkout通过63项CPU检查；`1166d12`的公有仓库CI已通过。

## 2026-10-08：固定容量与共同函数的shared行距对照

源码 `56db27e` 在同一非模板kernel中固定4160个shared元素，运行时选择行距64/65。38-case完整检查通过；后续十进程覆盖五个大shape，每个shape在每个位置出现两次、两种pitch先后各一次，100个确认case全部通过。五个shape的全部进程均观察到pitch65更快，进程内中位比值范围约1.079×–1.189×。

两份trace各110个kernel，函数名相同，都报告13个寄存器、16640字节static shared；缺省57-case及旧记录协议保留。这个对照减少了第一轮容量与模板实例不同的混杂，但shared活跃地址仍会变化，不能识别bank映射，也不能把新旧实现差异唯一归因于容量或occupancy。

[同一机制页的后继章节](../wiki/transpose.md#后继同一个函数固定16640字节容量) · [独立结果与1400个计时批次](../data/results/20261008-shared-pitch.json)。所有十三个设备进程及profiled应用已退出；仍不向open-cake-ir提升。源代码固定提交在独立checkout通过66项CPU测试，`820fc44`公有仓库CI已通过。

## 2026-10-08：动态shared请求与行距的分解对照

源码`dd20525`新增共同dynamic kernel的三个配置：pitch64请求16,384/16,640B，以及pitch65请求16,640B。57-case扫描后，十进程按预先固定的A_first/B/C/A_last或A_first/C/B/A_last复验200case，三份独立trace再验3case；260case全部正确，保存2600个原始批次。

固定pitch的A/B比较在五个shape中的四个稳定支持较小请求更快，第五个4095×4097的比值跨1，不能写统一容量规则。固定请求的B/C比较则五shape各十进程都支持pitch65更快。相同配置的前后端点、每个端点分别对B的比值全部保留，没有筛去漂移大的进程。

[机制页的后继对照](../wiki/transpose.md#再后继同一函数的动态shared请求量) · [完整结果](../data/results/20261008-dynamic-shared.json)。三个trace同名、各110kernel，均报告13regs/static0，请求对应的dynamic报告值为16384/16640；重编译标志均false。仍未识别物理分配粒度或实际occupancy。

全部14个设备worker及3个profiled应用已释放。固定源码在独立checkout通过70项CPU检查，并完成本机MXCC编译和host非法组合负对照；没有向open-cake-ir提升。`a61ea94`固定容量行距结果已按用户明确授权发布，其公有仓库CI已通过；后续同等验证的增量继续发布。

同时核对官方C++指南3.5.3.x的shuffle、integer reduction与同步契约，留存主文和活动版本选择器；当时这些接口尚未在本库完成C550实测，接口依据与后继范围见[工具链页](toolchain.md#64-lane-collective-的接口与实测范围)。

## 2026-10-08：完整wave64的shuffle与整数归约

源码`d01ef59`增加20个边界case：64/128物理线程、逻辑n跨31/32/33、63/64/65和第二wave的95/96/97等边界，另含n0。每个物理线程都执行完整64-bit mask的8种直接shuffle查询与signed-int求和，保存全部9个结果，尾部提供0而不提前退出。

正序20case、独立进程逆序20case和两份单case trace全量通过，累计39744个payload、2688个guards及420个原始事件批次。完整128thread时两个wave的和分别为2080和6176；width32的上半组及第二wave按各自子组寻源，source32/63的取模行为与文档及本机头文件一致。逻辑n65时，补零的thread127仍得到归约值65，说明只检查前n个thread会漏掉有效结果。

[新机制页](../wiki/wave-collectives.md) · [含全部实际整数输出的结果](../data/results/20261008-wave-collectives.json)。两个trace各110个kernel，block64/128分别可见，均报告16regs、shared/private0，重编译标志均false。源helper存在不证明单条硬件指令，保留的完整kernel时间不作intrinsic延迟或吞吐结论。

四个设备worker及两个profiled应用均已退出并验证释放。独立干净源码通过83项CPU检查，本机MXCC编译通过；无设备可见的block32/n32及block64/n65负对照在host拒绝。仍不向open-cake-ir提升。此前`9a23cb8`发布的公有仓库CI已通过。

## 2026-10-08：完整typed mask的SDK重载边界

源码`715e877`保留默认九通道探针，另增加mask-types双通道模式。编译验证unsigned32、unsigned long64、uint64_t类型身份、两个字面量类型和SDK的MACA_HALF_WARP_SIZE=32；两个typed变量直接传原生重载，没有公共wide-mask包装，也不执行截断的64-bit局部mask。

新模式正序20case、逆序20case与2个trace case全部通过；完整128thread的64-bit通道返回2080/6176，32-bit兼容通道返回528/1552/2576/3600。n65/thread127为[65,0]。这是两个API合同的分组差异，physical wave仍64，mask类型与数值都不同，不发布type-only因果或速度比。

[同一机制页的后继章节](../wiki/wave-collectives.md#后继两种完整typed-mask选择不同归约入口) · [所有实际输出与原始批次](../data/results/20261008-mask-overloads.json)。另有默认mode0的20case完整回归；合计62case、27840payload、3968guards和620batches通过。两个新模式trace各110kernel，16regs/shared0/private0，重编译标志均false。

全部5个device worker及2个profiled应用已退出。干净源码89项CPU检查通过，两个模式本机编译及4个无设备host负对照通过。第一次摘要采集遗漏空目录，严格投影因此拒绝；后继只读归档保留远端原始空目录后通过，未改实验或补造目录状态。仍不向open-cake-ir提升。之前`1f6a50a`的公有仓库CI已通过。

## 2026-10-08：原生WMMA的精确性反例

系统Python没有torch/triton并不代表整机没有框架环境。只读核实了现有containerd容器的包元数据：Torch2.10.0、Triton3.6.0及maca-tile1.0.1均带metax3.8.0.4.c600u后缀；未导入框架或执行其GPU路径。这与本轮宿主MACA3.5.3环境分开记录，不继承执行资格。

源码`ef22b51`用当前SDK原生header的mxmaca::wmma执行FP16输入、float累加的16×16×16操作，多K块均匀循环并验证尾部padding。实际编译及两个host负对照通过，干净源码102项CPU检查通过；但设备首次12case中10case没有满足预先固定的exact合同，共177个数值不等，最大绝对残差4.76837158203125e-7。输入正确、输出全有限、guards完整。原逆序与性能trace门禁保持未通过，未启动。

独立诊断用同一留存binary和输入重复K16两次、K64一次，另取K16资源trace，完整输出位均与首次对应case一致；原strict checker继续返回失败。对全部16个实际case的独立诊断包含32768个input halfwords、4096个C值、2048个guards，共277个不等值。逐步正确舍入的FP32 CPU replay仍精确匹配整数dot/256参考，普通求和重排不足以解释差异。

[数值诊断页](../wiki/wmma-exactness.md) · [保留failed状态的完整记录](../data/results/20261008-wmma-exact-diagnostic.json)。官方WMMA章节未给出严格IEEE逐步FP32舍入契约；原因未定位，不能直接称硬件缺陷。记录为device-correctness、passed=false、performance_accepted=false。索引新增明确的正确性诊断入口，原local-measurement性能门槛仍要求passed=true，oracle和容差未改。

五个device worker及一个profiled应用已退出并验证释放。诊断trace110kernel、28regs/shared0/private0、recompiled false110；没有由此推导算术机制或性能。本轮不向open-cake-ir提升。此前`4aa7d4a`公有仓库CI已通过。

## 2026-10-08：WMMA与scalar源码的设备输入控制

后继源码`403a74a`保留原WMMA函数体和严格oracle，增加使用同A/B设备分配的scalar-source FP32实现，并保存计算前、两实现之间、计算后三次完整输入回读。每个实现独立C及guards，输出立即按wmma/scalar角色保存，顺序不改变文件归属。源码体相同不作为二进制相同的证明。

12个正序/WMMA-first、12个逆序/scalar-first及1个K16配对trace，共25个logical case。153600个snapshot halfwords都与prepared/fixed输入相等；scalar的25份输出全部精确，WMMA仍有384个不等值（两遍各177，trace30）。新默认mode0的12case另作回归；全部37份WMMA输出逐word与旧ef22对应case相同。没有用scalar通过替换WMMA失败。

[控制页](../wiki/wmma-exactness.md#后继设备输入快照与标量fp32源码对照) · [全部输入回读与两路输出](../data/results/20261008-wmma-scalar-control.json)。配对trace按真实函数名分成两个110-event组，各自删除10次预热；报告WMMA block64/28regs、scalar block256/36regs，两组shared/private0、recompiled false110。计时只保留追溯，无性能接受或速度比。

这把排查范围缩小到本次插桩的WMMA路径，但三个捕获边界不证明kernel内部瞬态输入，仍不能唯一归责硬件。干净源码121项CPU检查、双模式本机编译和三个无设备host负对照通过；原先edf9857的软件CI通过不改变WMMA数值失败。四个worker及一个profiled应用已退出并验证释放，未向open-cake-ir提升。

## 下一轮问题

- 缩小逻辑M/N及K前缀，保持同一完整16×16物理tile、输入公式与scalar控制，寻找残差出现条件；首个witness不等于已证明的最小反例。
- 检查实际生成代码和WMMA路径的加载/算术实现；不从source-level scalar名称或API fragment类型推定ISA行为。
- 原失败、精确oracle和非性能诊断状态保留。新的容器Triton路线仍仅完成包身份调查，不能承接当前native路线的资格。
