# 来源与复用范围

本库为 MetaX C550 的 kernel 开发与优化建立可检索知识。硬件身份、MACA API、编译器行为和性能结论各自需要本目标的来源；参考其他 kernel wiki 的组织方法不建立 C550 的硬件事实。

## 已检查的参考库

本次检查日期为 2026-10-07，固定以下版本；这是读取日期，不是来源发布日期。

| 参考 | 检查版本 | 沿用的组织方法 |
|---|---|---|
| [qhy991/bw1100-kernelwiki](https://github.com/qhy991/bw1100-kernelwiki) | `main`，[`6b4a6de8cdece72fea1ac8dd8fee0a104e491a94`](https://github.com/qhy991/bw1100-kernelwiki/tree/6b4a6de8cdece72fea1ac8dd8fee0a104e491a94) | 先记录来源，再按代价、改写、条件、退化与本机范围组织机制；以 `evidence_scope` 区分环境、编译、数值、配对计时和无效实验 |
| [qhy991/metal-kernelwiki](https://github.com/qhy991/metal-kernelwiki) | `main`，[`30d36da6c1d2543372793cbf020863a5f26dce85`](https://github.com/qhy991/metal-kernelwiki/tree/30d36da6c1d2543372793cbf020863a5f26dce85) | 单一页面与来源目录、离线查询、薄 skill 入口；公开知识与仓库外原始实验分离；明确文档、推断与本地测量的边界 |

BW1100 库是私有资料，链接可能要求访问权限。Metal 库在检查时为公开仓库，其来源说明引用的是更早的 BW1100 提交 `52ae9a1`；本表记录本次实际检查的版本，二者不可混为同一次检查。

具体借鉴点及出处：

- [BW1100 维护规范](https://github.com/qhy991/bw1100-kernelwiki/blob/6b4a6de8cdece72fea1ac8dd8fee0a104e491a94/MAINTENANCE.md)要求说明优化的因果关系、适用条件和反例，保留被修正的观察。
- [BW1100 证据校验](https://github.com/qhy991/bw1100-kernelwiki/blob/6b4a6de8cdece72fea1ac8dd8fee0a104e491a94/scripts/validate-evidence.py)区分证据范围；编译和协议记录不能承担性能声明，配对记录必须有分母。
- [BW1100 知识测试](https://github.com/qhy991/bw1100-kernelwiki/blob/6b4a6de8cdece72fea1ac8dd8fee0a104e491a94/tests/test_knowledge.py)验证检索、过滤、引用和证据约束，不把结构测试解释为 GPU 验证。
- [Metal 维护规范](https://github.com/qhy991/metal-kernelwiki/blob/30d36da6c1d2543372793cbf020863a5f26dce85/MAINTENANCE.md)以逻辑 `artifact_ref` 引用仓库外实验，并说明外部读者无法仅靠页面复验原运行。
- [Metal 测量方法](https://github.com/qhy991/metal-kernelwiki/blob/30d36da6c1d2543372793cbf020863a5f26dce85/wiki/measurement.md)分离计时范围、正确性、原始样本和 profiler 观察；清理分配器不等于刷新硬件缓存。
- [Metal 来源说明](https://github.com/qhy991/metal-kernelwiki/blob/30d36da6c1d2543372793cbf020863a5f26dce85/PROVENANCE.md)保留复用范围与许可边界，不把引用链接当作整份内容的再分发许可。

## 本库自己的证据

上游文档和源码链接只支持其明示内容。一个文档声明、一个 API 返回值、一个成功编译和一次设备运行是不同层次的证据。本库不将其他厂商的 wave/warp 宽度、缓存、指令、计时或资源限制迁移为 MetaX 属性；兼容 API 的名称也不证明底层硬件相同。

每项本地结果应能定位其运行 ID、执行源文件、环境记录、输入与 oracle、失败、原始计时样本和测量范围。原始记录由仓库外的实验目录持有，知识页保存派生解释和来源引用。公开页面只使用 `运行ID/相对文件` 形式的逻辑引用；机器地址、账户、密钥及私有路径映射不进入公开内容。原始记录未随仓库发布时必须明确说明，不声称第三方已经可以独立复验。

本文的参考检查不包含在参考库的设备上重新运行实验。这里没有复制 BW1100 或 Apple 的实验结果、知识正文或检索实现，也没有用它们的 GPU 数值设定 C550 的默认参数。若后续直接复用代码，应另记具体文件、固定版本和实际许可。引用不补造第三方许可。

本库的测量与维护约定见[实验与知识维护方法](methodology.md)。资料数量不是覆盖度；新的结论需要新的可定位证据，既有失败不能被成功重试覆盖。
