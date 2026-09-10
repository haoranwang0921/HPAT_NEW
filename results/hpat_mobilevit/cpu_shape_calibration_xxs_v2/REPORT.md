# 逐算子、逐形状校准：本机先导微基准

已完成本机 CPU 微基准，尚未完成手机 INT8 NPU 校准。未修改 D0 或 HPAT 成本参数。

环境：AMD Ryzen 9 7940HX；Windows；PyTorch 2.10.0+cpu；FP32；单线程；固定随机种子20260906。输入为XXS 192×192 batch1 trace的形状，随机连续张量，非原模型张量步长/权重重放。每个形状三轮blocked_autorange，每轮至少0.1秒，含自动预热；计入函数调用及输出分配，不计输入构造。v1与测试进程重叠，仅作冒烟；v2运行时未主动运行其他校验作业。

覆盖49种形状、90次算子调用、210041600 MAC，即trace中全部矩阵类算子的MAC。包含普通卷积、深度卷积、Linear、注意力BMM；不含Softmax、LayerNorm等非矩阵算子。

有效FP32 GOPS范围：普通卷积42.61–129.47；深度卷积1.73–23.16；Linear20.49–115.37；BMM9.40–90.43。三轮中位数的最大/最小比超过1.2的形状共4种，需增加重复及独立会话确认。

按trace调用次数加权的独立算子中位数总和为5.495791 ms，不是整网实测时间。当前D0矩阵公式相应总和0.0525104 ms；CPU FP32与抽象NPU INT8不同，不能据此给D0乘一个比例或认定低估倍数。

下一步：在明确设备、精度、布局、线程、编译后端及融合边界后，采集同类逐形状表；用独立测量会话检验查表误差及未见形状误差，再用整网实测闭合。实测算子耗时包含内存与运行时成本，不能未经拆分直接叠加现有外存账本。手机校准还需要目标设备执行工具/原始计时记录。

文件：shape_timings.csv为形状测量表，raw_measurements.json记录形状签名、对应op_id和全部分块计时，manifest.json保存环境和输入/脚本哈希。测量通过输出形状及有限值检查；现有49项回归测试通过（非此测量表的硬件准确性证明）。

复现：`python -m experiments.hpat_mobilevit.benchmark_operator_shapes --trace results/hpat_mobilevit/trace_v1/xxs/operator_trace.jsonl --output results/hpat_mobilevit/cpu_shape_calibration_xxs_v3 --threads 1`
