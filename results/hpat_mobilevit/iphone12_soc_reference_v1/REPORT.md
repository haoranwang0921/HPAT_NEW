# iPhone 12 延迟参考模型：第一版试运行

日期：2026-09-10。状态：H1/H2/H3 冻结完成；独立电子参考后端、测试及论文延迟锚点拟合完成；**真实 SoC 校准尚未完成，不能替代公平架构对照中的 D0**。

## 1. 冻结范围

`../h123_frozen_20260910_v1/snapshot.zip` 保存207个源码、配置、trace和结果文件，manifest.json逐文件记录SHA-256。
冻结的是最近Phase 2的XXS/192 H1、H2、H3：32×32、4核、10GHz、25MiB权重缓存、64GB/s外存及原电子残余成本，不是最近尚未执行的1MiB/512GB/s三档profile方案。
归档SHA-256：72424dc852def6a44e065c3b53d2362e82677e0e3688fd06582482da523947d3。
新模型运行前后复核一致。没有修改streaming.py、backends.py、mobile_profile.py、joint_sim/electronic_backend.py或H1/H2/H3输出。

## 2. 论文测量依据

原文：[MobileViT，arXiv v2，表11及§4.3](https://arxiv.org/html/2110.02178v2)。
对应XS规模（约2.3M参数），256×256、batch=1，各阶段patch=2；预训练全精度模型转换为CoreML，论文报告100次迭代平均延迟。

| 图执行目标 | 原论文时延 | 本次锚点拟合结果 |
|---|---:|---:|
| iPhone 12 CPU | 17.86 ms | 17.86 ms |
| iPhone 12 Neural Engine | 7.28 ms | 7.28 ms |

原文没有由这些数值给出可用的逐算子耗时、完整融合图、各算子落在哪个单元、内部实际混合精度、可配对功耗/能量。**本次energy及power返回null，不填0，也不借用原D0能耗当iPhone数据。**

## 3. 代码分离与模型等级

新增 `experiments/hpat_mobilevit/soc_reference.py`，由独立 `run_soc_reference.py` 调用，不接入任何混合H路径。
类别包括dense、depthwise、attention、vector、layout、metadata；参数为图执行目标的有效吞吐先验、逻辑带宽、类别效率、调用开销和一个全局运行时校准系数。

每算子基础时间：dispatch + max(compute proxy, logical-memory proxy)。按trace依赖和程序顺序串行安排，再乘该执行目标的全局校准系数。
逐算子输出保留原始计算/逻辑访存/调用先验与最终时长，避免把这些分量误相加。

这属于**可解释特征 + 全图经验拟合的分析模型**，不是A14微架构、周期精确、CoreML真实调度或真实CPU/ANE异构并发仿真。logical bytes是张量访问代理，不是HBM/LPDDR真实字节。没有使用光子切块、signed双pass或每128B权重行的光子供数政策来模拟手机。

| 先验 | CPU目标 | ANE目标 | 证据等级 |
|---|---:|---:|---|
| dense服务速率 | 100 Gop/s | 1 Top/s | 未校准建模先验，不是Apple规格 |
| 逻辑访存速率 | 20 GB/s | 32 GB/s | 未校准先验，不是A14外存实测 |
| depthwise/attention/vector相对效率 | 0.1/0.25/0.1 | 相同 | 未校准 |
| 每非metadata算子调用先验 | 2 us | 2 us | 未校准，非CoreML测量 |
| 逻辑张量计数位宽 | 32 bit | 32 bit | 对应FP32捕获的代理，不证明ANE内部FP32 |

CPU全局系数0.3431411，ANE全局系数0.5800702。未经拟合的XS256分别为52.048559ms、12.550204ms。拟合命中只表明一个自由参数拟合了一个锚点，**不能声称模型精度100%或逐算子完成校准**。

## 4. 新trace及未关闭的协议差异

独立生成 `../iphone12_trace256_v1/xs/`：timm mobilevit_xs、输入256×256、随机种子20260706、pretrained=false、成本字段32bit。580个ATen算子、2,317,848参数，捕获执行输出与非捕获执行一致；本机检查三个MobileViT块patch均为(2,2)。未下载官方权重。

新trace矩阵类统计：Conv2d 607,440,896 MAC；Linear 301,194,240；attention MatMul 117,276,672；合计1,025,911,808 MAC。
原表列出0.7G FLOPs。两者的统计口径、统计分辨率、实际导出图是否相同尚未确认：不能仅凭模型名和参数量相近宣布一致，也不能简单用0.7/1.026缩放trace。
原trace的reference_comparison对应旧XS224，故新256 trace显示matching=false是预期的分辨率差异，不代表已通过原论文图的匹配验证。
原论文使用预训练CoreML模型，本次只用timm随机权重未融合ATen形状；内部融合、布局、精度及fallback尚未重现。因此当前拟合应称**跨实现的数值锚定试验**，不是原实验复现。

## 5. 外推结果不得用于优势宣称

aggregate.csv中的XXS192、XS224、S256行均标记unvalidated_extrapolation，没有对应原论文精确实测标签。
identifiability.csv展示：三套不同带宽/调用先验都能拟合XS256=7.28ms，但XXS192预测分别约1.936、2.079、2.452ms。这不是统计置信区间，而是参数不可辨识示例。

不能把此处外推的SoC延迟与冻结HPAT结果直接相除作为加速比：不同内部精度、实现、调度边界；H路径电子残余仍使用冻结的旧成本。保留旧D0作为原对照，独立命名新基线D0_SoC_reference。

## 6. 验证与下一步

141 tests passed，3条第三方弃用警告；测试覆盖协议拒绝（XS224不能校准到XS256锚点）、依赖顺序、未知算子拒绝、时间闭合、能耗未知和单点拟合标签。H123前后哈希未变。
verification.json的valid仅指这些代码/记账检查。硬件验证门禁仍为未通过，见validation_gates.json。

下一步需要先对齐原版Apple MobileViT的精确计算图/运算统计与CoreML转换设置，再在至少一个独立设备测量点验证误差；逐算子校准需逐算子测量或编译器profile。功耗需独立实测或来源清楚的能量模型，不能由两条延迟反推。

复现命令：

```
python -B -m experiments.hpat_mobilevit.trace --output results/hpat_mobilevit/iphone12_trace256_v2 --variant xs --resolution 256 --bits 32
python -B -m experiments.hpat_mobilevit.run_soc_reference --output results/hpat_mobilevit/iphone12_soc_reference_v2 --trace results/hpat_mobilevit/iphone12_trace256_v2/xs/operator_trace.jsonl --freeze results/hpat_mobilevit/h123_frozen_20260910_v1
python -B -m pytest joint_sim/tests experiments/hpat_mobilevit/tests -q -p no:cacheprovider
```
