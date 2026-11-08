# SmolVLA / Jetson Orin 深度 profiling 与 250 ms 优化目标

日期：2026-11-05。本文补充 [profiling 记录](smolvla-cuda-profiling.md) 和 [CUDA 优化执行计划](smolvla-cuda-optimization-plan.md)，不覆盖其中不同实验条件下的历史数据。

> 本轮只进行测量、临时副本插桩与方案设计，不修改生产推理实现。用户提供的 LeRobot 约 250 ms 是待对齐的参考目标，不是本仓库已复现的 LeRobot 基线。历史约 410 ms 与本轮结果不能直接计算代码加速比。

## 1. 对齐比较口径

两套框架应分别报告以下两个范围，不能交叉比较：

1. **完整 policy**：三路原始 CPU RGB、归一化 state、文本和固定 noise 已准备好；从开始处理输入到 GPU 工作完成、CPU 得到完整动作 chunk。不含模型加载、文件读取、网络排队、机器人执行。包含 resize/pad/normalize、tokenization、H2D、视觉、Prefix、全部十步 Expert、D2H。
2. **准备好 GPU 输入后的模型计算**：排除 CPU 输入预处理，但仍必须完成真正的 forward 和十步 sample，并在外层同步后停止计时。当前 C++ policy 的 `total_ms` 属于第一类，不能与第二类直接相减。

必须核对的工作负载：

| 项目 | 当前 C++ 基准 | LeRobot 对比必须记录 |
|---|---|---|
| Batch / 相机 | batch 1，三路 256×256 RGB | 同相机数量、顺序、像素与有效标志 |
| 视觉实际计算分辨率 | 每路 resize/pad 到 512×512 | 不能把原始 256×256 当作 ViT 输入分辨率 |
| 视觉 token | 每路内部 1024，projector 后 64 | 区分 attention token 与 Prefix token |
| 视觉层 | 12 层 | 不能把训练时 freeze 当作跳过 forward |
| Prefix | 16 层，241 个分配位置；当前文本 13 个有效语言 token、206 个有效 Prefix token；末层仅 K/V | 同 tokenizer、实际有效 token、padding、mask、position |
| 文本 | `Pick up the red cube and place it in the bowl.` | 同一句；旧 Python fixture 使用不同的 block/tray 文本，不能直接重生成替代 |
| State / action | state 6，内部补齐 32；动作 chunk 50×32，公开输出 50×6 | 相同输入归一化、输出截取、统计量和固定初始 noise |
| Expert | 16 层，10 次采样 | 不允许减少步数或误将单步时间作为总时间 |
| 缓存 | 图/权重复用，每次重算三个相机和 Prefix | 不能以 `select_action()` 动作队列命中代替生成新 chunk |
| 精度 | GGUF 混合 F16/F32；显式 BF16 舍入边界；FA OFF | 权重 dtype、运行时 dtype、AMP、TF32、SDPA backend、compile 均需明确 |
| 计时 | 同进程重复；首轮单列；完成式 wall time | 相同 warm-up，显式 CUDA 完成边界，报告 mean/median/p95 |
| 硬件状态 | Orin、MAXN、未锁频 | 功率模式、CPU/GPU/EMC 频率、温度、后台负载、版本 |

本机存在 LeRobot SmolVLA 的 HF checkpoint 缓存，但发现的 Python 环境没有可运行的 `lerobot` / `transformers` / `numpy` / `safetensors` 依赖组合，也没有完整的 backbone 资产。缓存 checkpoint 与本次 GGUF 的导出版本身份仍需核对。**本轮不安装包、不下载资产，因此不声称已经完成 LeRobot 的同机对照。** checkpoint 的 BF16 存储不能证明运行时也使用 BF16，`use_amp=false` 也不能单独确定所有算子的计算精度。

## 2. 测量方法与边界

产物根目录：`tmp/smolvla-cuda-20261104-1219/comprehensive-20261105/`。

- 未插桩测量使用已验收默认 CUDA 二进制，清除继承的 `SMOLVLA_*` 和 `GGML_CUDA_*` 调优变量，再仅设置重复次数、临时目录和 CUDA cache 路径。模型、输入及 noise 沿用原 CUDA fixture。
- `scripts/baseline.py`：三次进程运行，每次 31 次 policy；每次首轮单列。两次无 telemetry 的 60 个 warm 样本用于主统计；中间一次的 30 个 warm 样本单独记录 tegrastats 和只读 CPU/GPU 频率采样，不混入主统计。
- 最初一批测量与临时 profiling 编译重叠，仅作为探索记录，不能用作隔离基线。没有任何生产实现变更时出现延迟变化，也不能解释成代码优化。
- `scripts/capture.py`：单独的源码副本，原有 NVTX 加视觉细分范围；Nsight Systems trace CUDA / NVTX / OS runtime，展开 CUDA Graph 节点。7 次 policy，排除首轮的 6 次结果完整保留；发现第二次仍 capture/instantiate 26 个图后，另报排除前两次的 5 次 steady 数据。两者不混用。
- `scripts/capture_host.py`：只 trace NVTX，不采 CUDA / OSRT / CPU sampling；31 次 policy，保留排除首轮的 30 次和排除前两次的 29 次统计，用来量化预处理/staging 并检查逐节点 tracing 干扰。两类 capture 的动作均实际与保存的默认 CUDA 动作逐位一致。
- `scripts/analyze.py`：GPU activity 必须完整落在同步阶段 NVTX 范围中才归属该阶段；不以不唯一的 correlation ID 重复 join。分别报告 kernel sum、kernel+memcpy active interval union、wall、kernel/API/copy 数量和字节。
- `expert.step` 属于 Expert 的子范围，视觉 detail 属于 camera 子范围，父子不可相加。CUDA API 的同步时间包含有用的 GPU 等待，不能与 kernel 时间相加，也不能全部称为可消除开销。`wall - GPU active union` 包含主机处理、提交间隙、driver 和插桩影响，不等于纯 CPU 时间。
- Orin 使用共享物理内存；H2D/D2H 字节数不代表 PCIe 流量，logical tensor bytes 也不等于实测 DRAM 流量。未取得硬件计数器时不作带宽饱和/occupancy 瓶颈结论。

### 硬件计数器限制

Nsight Compute 对一个代表性 vision softmax kernel 的探测失败：`The user does not have permission to access NVIDIA GPU Performance Counters`，目标进程随后被 profiler 终止，错误码 9。最初带模板维度的 kernel 过滤器还曾无匹配；退出码 0 不等于取得数据。保留 `ncu-probe.json` 和 `logs/ncu-softmax.csv`，不报告虚构的 occupancy / DRAM throughput。未调整设备权限、功率、时钟或 sysfs。

## 3. 结构成本：用于定位，不代替 trace

估算按一次 FMA=2 FLOPs，仅统计主要 dense math：三路视觉约 646 GF，Prefix 约 74.72 GF，十步 Expert 约 104.76 GF，另有 cross-KV 预计算等。总量约 0.83 TF；这不是实测吞吐，也不构成延迟下限。

视觉非 FA 路径每层 score 为 `12×1024×1024×4 = 48 MiB`，每次请求 36 个 image-layer。仅 score 写入约 1.812 GB，score/probability 的若干读写逻辑量约 7.25 GB；没有计数器时不能把它当作实际 DRAM 事务量。当前走通用 Idefics3 ViT，不能假设其他模型的专用视觉 builder 已应用。

Prefix 单层 resident 仍在 `src/models/smolvla.cpp:1825` 至 `:1837` 逐层上传 hidden/positions/mask、读回 hidden/K/V/validation。全链 resident 与 device KV bridge 有明确的主机往返消除对象，但其收益上限必须受当前 Prefix 阶段时长约束。

显式 BF16 round-trip、finite probe、GQA expand 和 layout copy 是后续重点候选，而非已经证明的主要耗时。Prefix GQA 见 `src/models/smolvla.cpp:1784`，Expert 见 `:1147`；query head h 对应 KV head `h/3`，不是 `h%5`。

## 4. 细粒度优化实验及门禁

以下为实验顺序框架；实际先后应服从本轮 warm trace 的耗时排序。收益必须以单项未插桩 A/B 测得，不能累加多个相互覆盖的理论上限。

### E1：视觉 softmax / attention 中间张量

- 对象：`ggml/src/ggml-cuda/softmax.cu` 的 1024-column softmax，以及视觉 QK→softmax→PV 的 layout/转换。
- 先测 36 次视觉 softmax 总时长与 shape、launch 参数；若占比高，做固定 shape 的布局/访存实验，保留 scale、mask、exp 和 reduction 语义。
- 线程块/规约顺序变化也可能改变浮点结果，不能因“同一公式”自动归入严格等价。先逐元素 vision embedding 比较，再跑 policy。
- 若 softmax 时间下降却被 QK/PV/layout 回退抵消，则停止，不仅报告 microbenchmark 加速。
- **现有 FA 已在此前严格数值资格验证失败，仍关闭。** 在线 softmax、SDPA/FA 等改变规约顺序的方案只能进入独立精度实验，不能绕过失败记录。

### E2：GEMM 共享 operand conversion 与布局

- 对象：共享 norm 的 Q/K/V、共享 MLP 输入的 gate/up，及 F32↔F16 `convert_unary`。先从 trace 核实每组转换次数和持续时间，再考虑显式共享一次转换、保留原 GEMM accumulator/output 类型。
- 分开测权重常驻转换、activation 转换、output 转换；防止把本来无法共享的不同 tensor 混为一谈。权重转换需核对持久内存和模型生命周期。
- 不要求增加 fused QKV 大矩阵；合并 GEMM 可能改变算法与舍入，必须重新资格验证。
- 对转换节点的收益上限只取相关类别实测时间，不把 GEMM 本体全部算作可消除。

### E3：精确 BF16 round / finite probe 融合

- 对象：`src/models/smolvla.cpp:1107` 的 round，`:1117` 的 finite probe，以及 Prefix 对应节点。
- 候选为单个 F32→BF16→F32 round kernel，或与原有逐元素算子融合，必须保留每一个中间 BF16 舍入点、NaN/Inf 行为及原算子计算顺序。
- finite 检查可研究与 producer/round 融合，但必须观察舍入前后所有原先检查点，包括 F32 有限大值转 BF16 溢出；不能把 `0*x` 常量折叠为零。probe 的诊断结果不可进入模型数值路径。
- 逐一减少保守 OUTPUT pin 前必须证明 allocator lifetime 和 view root 安全，不能为了峰值内存直接删除 retain。

### E4：Prefix 全链与 Prefix→Expert device bridge

- 先复用 Expert 已验收的链式图经验，将 15 个完整 Prefix 层加末层 K/V-only 接成固定图；保留 public host K/V API，内部 policy 使用独立 device path。
- 第二步才桥接到 Expert K/V，并处理 cross-KV 一次/请求的投影与静态输入刷新；不要把已经每请求执行一次的 cross-KV 误当成每步重复优化。
- 每层 K/V 逐位对照；同句柄 shape/mask/content 改变、A→B→A、失败恢复、CPU 路径不变。缓存先 graph/scheduler 再 weights/backend 销毁。
- 必须记录单配置进程峰值和 scheduler allocation；模式切换时可能同时持有多个缓存，不能只用单个 scheduler buffer 代替整体显存。

### E5：紧凑 GQA / 静态 K/V layout

- 避免显式 repeat 和重复 V transpose，考虑直接读取紧凑 5-head KV；每个 query head 的 KV 映射保持相邻三头分组。
- 静态 prefix 部分的布局可每请求预处理一次；self-attention 的 suffix K/V 每步变化，不能跨十步错误复用。
- 任意 native-GQA kernel 必须维持 attention 精度、scale=0.125、NEOX RoPE 和 finite-mask 行为；fully masked 行当前为有限 `-FLT_MAX` 对应的 uniform softmax，不能改成零行。

### E6：视觉 staging 与 camera batching

- 先用 detail NVTX 判定 resize/normalize、图像深拷贝、HWC→CHW staging、upload、readback 的实际上限。先复用容量/去冗余拷贝，不假设 CPU 预处理是最大瓶颈。
- 三相机 batching 需 batch-aware pixel unshuffle/projector，不能直接改变张量维度后默认正确；需要逐相机 embedding、不同内容顺序、原始 aspect ratio 与失败恢复测试。
- 更大 batch 可能改变 GEMM 算法、舍入和峰值内存；没有未插桩端到端收益或不满足逐位 gate 时不默认启用。

### E7：GEMM 精度/算法独立试验轨

- `scripts/gemm_matrix.py` 将默认前后对照与 `GGML_CUDA_F16_F32_OUT=1`、其与 LT autotune 组合、强制 BF16/F16 compute 分开测量；记录延迟、finite、不同元素数、max abs、RMSE 和 cosine。
- 强制 compute 类型可能覆盖 `GGML_PREC_F32`，F16/F32-out 改变累加/输出行为；autotune 改变算法也可能改变数值。它们均**不是已经验收的严格等价调优选项**。
- 单组 300 动作即使完全相同，也不能替代三路 embedding、六组 policy、mask/极端值/恢复回归及任务级准确率评估。所有诊断结果默认 `adopted=false`，不据此更改生产默认。

## 5. 验收与 250 ms 里程碑

1. 先得到无编译/无其他本会话 GPU 测试重叠的基线与 trace，并明确 DVFS、温度、采样数量和 p95。若前后 control 漂移显著，交错重测，不给出伪精确加速比。
2. 每次只改一个候选：microbenchmark → component → 六组 policy → 同句柄内容/mask/shape/异常恢复 → CPU/default backend/public API/真实 HTTP 回归 → NVTX OFF 交错 A/B。严格路线保持同 backend 逐位等价，不放宽原门禁。
3. 阶段预算用 `T_vision + T_prepare + T_prefix + T_expert + T_other ≤ 250 ms` 表示。实际基线确认后计算缺口；优先从占比最高且存在可测冗余的类别入手。预算是工程目标，不是预测。
4. 对每个候选列出“相关 kernel/copy 时间上限、实测 A/B 收益、额外持久内存、精度资格”。不同候选共享转换/布局成本时不能把收益简单相加。
5. 若严格路线的累计实测仍不足以达到 250 ms，应明确报告剩余差额，并把 mixed precision / FA / compile 等放入需要独立准确率授权和评估的路线；不减少摄像头/分辨率/采样步数来冒充同工作量优化。

## 6. 当前结果状态

### 6.1 未插桩完整 policy 基线

构建完成后顺序执行，无本会话编译或 GPU 实验重叠；不是全机独占或锁频保证。两次无 telemetry 进程各 31 次、各剔除首轮，共 60 个 warm 样本：

| 阶段 | mean ms | median ms | p95 ms | min / max ms |
|---|---:|---:|---:|---:|
| 三路 ViT | 244.015 | 244.361 | 248.779 | 210.154 / 263.505 |
| Prefix prepare | 1.978 | 1.977 | 2.077 | 0.827 / 2.448 |
| Prefix prefill | 45.826 | 44.442 | 47.574 | 35.110 / 78.310 |
| 十步 Expert | 109.695 | 109.691 | 111.410 | 98.881 / 112.198 |
| 完整 policy | **401.915** | **401.017** | **419.345** | **355.377 / 443.557** |

百分位采用 nearest rank。每阶段的 p95 不可相加。三个进程首轮 total 为 692.293 / 708.417 / 714.146 ms；第二次单独 telemetry 进程 warm total mean=402.331 ms，不混入上述 60 样本。三个输出文件均与原 CUDA 基线逐位一致。

与 250 ms 目标相比，mean 缺口 **151.915 ms**，需减少约 **37.8%**，即约 **1.608×** 端到端加速；若目标是 p95≤250 ms，则缺口为 169.345 ms。这里不是相对 LeRobot 的实测差距。

探索组编译重叠时 total mean=343.363 ms，独立组=401.915 ms，生产代码完全相同。探索组 telemetry 曾记录 CPU 约 2201 MHz；独立组 139 个只读采样中，各 CPU policy 大多数处于 729600 kHz，GPU 有 124 个采样为 1300500000 Hz。采样包含加载/退出，不能等价为每个算子的执行频率，但说明不能忽略 CPU DVFS。MAXN 不等于 CPU/GPU 锁定最高频；不能把编译负载带来的频率状态当作优化方案，亦不能把两个均值的差额归给图缓存。原始两组分目录保留。

### 6.2 主机细分：NVTX-only 低干扰测量

30 个 warm 完整 policy mean=401.848 ms、median=400.245 ms，与未插桩主基线接近；这是单独运行，不与主基线逐样本配对，仍不称为完全零干扰。

| 三路视觉细分，按每次 policy 求和 | mean ms |
|---|---:|
| RGB resize/pad/normalize | **53.967** |
| image staging | 2.507 |
| batch image 深拷贝 | 1.709 |
| HWC→CHW staging | 2.743 |
| 像素 upload，含 API 开销 | 2.552 |
| 同步 graph compute | **179.791** |
| embedding readback | 0.503 |
| embedding finite 检查 | 0.335 |

以上 detail 为各自独立 sibling 范围，合计约 244.11 ms；camera 总和约 244.46 ms，余量为未细分调用/检查。不与下表的 GPU kernel 时间混算。CPU 预处理已经是明确的高优先级对象，而单纯去掉约数毫秒的 staging 不足以解释 152 ms 缺口。

Expert mean=109.947 ms；其中 cross-KV-precompute 子范围为 7.712 ms，十步 resident-chain 子范围合计为 101.782 ms。`expert.step` 是更外层采样步范围，首步包含静态输入刷新/cross-KV 工作，不能用首步时长乘十；也不能把外层与子范围相加。

### 6.3 逐节点 CUDA trace：瓶颈排序，不替代端到端基线

先排除首轮的 6 次 mean wall=513.263 ms。CUDA Graph 第二次推理仍有 26 次 instantiate/update；第三至第七次均为零。故下表使用第三至第七次的 **5 次 steady** 数据（`trace-analysis-stable.json`），对应 wall=502.714 ms，比未插桩主均值高约 25.1%。不从这些时长直接预测未插桩加速率。

| 阶段 | trace wall ms | kernel sum ms | GPU kernel+copy active union ms | kernels / policy |
|---|---:|---:|---:|---:|
| 三路视觉 | 257.463 | 184.378 | 185.562 | 1,257 |
| Prefix prefill | 58.969 | 25.547 | 27.893 | 1,483 |
| 十步 Expert | 183.643 | 90.166 | 92.295 | 16,128 |
| 完整 policy | 502.714 | 300.113 | 305.780 | **18,870** |

| kernel 类别 | 全 policy ms | 视觉 ms | Prefix ms | Expert ms |
|---|---:|---:|---:|---:|
| softmax | 87.306 | **80.066** | 1.634 | 5.605 |
| GEMM / matvec | 71.893 | 41.527 | 5.007 | 25.343 |
| GEMM operand/output convert | 38.152 | **28.248** | 3.124 | 6.780 |
| 显式 BF16 copy/cast | 30.671 | 0 | **9.821** | **20.851** |
| elementwise binary | 24.870 | 15.023 | 1.598 | 8.250 |
| layout copy / concat | 20.107 | 9.469 | 1.742 | 8.896 |
| scale | 6.257 | 0 | 0.770 | 5.486 |
| reductions | 4.585 | 0 | 0.370 | 4.216 |

类别按 demangled kernel 名归类，不是每个高层算子语义的完美映射；其余 activation/norm/RoPE 等见 JSON。视觉每请求 36 个 `soft_max_f32<true,1024,1024,float>`，约 **43.4% 视觉 kernel sum**，高于视觉 GEMM 总和；优先研究这一路径，不应继续以纯 host 构图为首要目标。BF16 copy/cast 为 Prefix 578 次、Expert 5620 次，共 6198 次；不是未计入 output norm 的源码估计 5580 次。所有类别为实际 kernel duration 求和，不说明占用率或带宽饱和。

每次完整 policy 有 259 条 memcpy activity、84,025,636 bytes：H2D=44,910,268 bytes，D2H=29,678,184 bytes，D2D=9,437,184 bytes。Prefix 单独有 110 次拷贝、42,060,348 bytes，Expert 有 136 次、22,165,160 bytes。全部 memcpy activity time sum 约 6.16 ms；单凭字节量不能声称去传输可省数十毫秒，但减少 API/同步边界仍值得独立测量。

每次 steady 记录 39 次 `cudaGraphLaunch`、295 次 `cudaStreamSynchronize`（包含 prepare 等阶段），而不是 39 个 kernel。包括 capture 的七次分别为 graph launch `11,39,39,39,39,39,39`，instantiate `2,26,0,0,0,0,0`。与旧三次合计 89 次 launch 的 trace 是不同采样窗口，不混算。六次 warm 的同步 API duration sum=316.158 ms，已包含 GPU 等待，**不是额外可删的 316 ms**。逐节点 tracing 对大 Expert 图提交影响明显；不能把其约 91 ms 的 wall-active 差额全部当成生产 CPU 开销。

### 6.4 GEMM 数值/性能诊断

每配置 11 次、排除首轮 10 个样本；顺序前后默认 control，无更改实现。动作比较仅限这一个固定 300-float fixture：

| 配置 | total mean / median ms | ViT mean ms | Expert mean ms | 不同动作值 | max abs | 处理 |
|---|---:|---:|---:|---:|---:|---|
| 默认 control-before | 405.218 / 400.364 | 245.333 | 110.176 | 0 | 0 | 对照 |
| F16/F32-out | 403.117 / 398.318 | 241.625 | 112.362 | 300 | 0.010543 | 不采纳 |
| F16/F32-out + LT autotune | 395.501 / 392.613 | 236.126 | 110.894 | 300 | 0.009535 | 不采纳 |
| 强制 BF16 compute | 506.804 / 493.003 | 280.720 | 158.617 | 300 | 0.007800 | 不采纳 |
| 强制 F16 compute | 453.913 / 448.995 | 287.389 | 115.468 | 300 | 0.008692 | 不采纳 |
| 默认 control-after | 408.585 / 400.816 | 245.007 | 110.952 | 0 | 0 | 对照 |

四个非默认配置输出均有限但均非逐位等价；即使 LT 组合单次诊断较快，也远不足以接近 250 ms，而且未过严格数值门禁，不能作为默认推荐。强制 BF16/F16 反而更慢，不能根据硬件支持 Tensor Core 就推断全局切换一定有效。完整 RMSE/cosine/逐轮时长在 `gemm-matrix.json`；本轮没有重复六组资格验证，因为单 fixture 已经不满足严格路线。

### 6.5 更新后的执行顺序、预算与停止条件

| 顺序 | 实验 | 实测依据 / 成本上限参照 | 首轮实验目标（非预测） | 停止/升级条件 |
|---|---|---|---|---|
| P0 | **E6a 精确 CPU 预处理并行/SIMD/容量复用** | NVTX-only preprocess 53.967 ms；源码 `src/jetson_pi_smolvla.cpp:96`，当前逐像素双线性 | 保留逐像素计算顺序，先试行并行与坐标预计算；争取降低约 15–30 ms | 若坐标/FMA 改变输出即拒绝；若低频下线程调度抵消收益则停止；GPU 预处理另做数值资格 |
| P0 | **E1 1024-column vision softmax** | 36 kernels、80.066 ms trace，是最大的单一 kernel 类型 | 先做严格等价 microbenchmark；20–35% 单类下降约为 16–28 ms trace 量级，仅作实验筛选 | reduction 改序导致 embedding 不同即不能进入严格默认；端到端无收益即停止 |
| P1 | **E2 vision conversion / layout / bias** | convert 28.248、layout 9.469、binary 15.023 ms trace | 共享可重复 conversion、消除可证明冗余的布局；先争取单项 ≥3 ms 端到端收益 | 不能把三类全量作为可省成本；融合需保持原加法/舍入 |
| P1 | **E3 BF16 round 与 probe 融合** | BF16 30.671 ms、scale+reduction 10.842 ms 全请求 trace | 先单个精确 round kernel，减少 launch/临时 buffer；再试 probe 融合 | 保持有限大值溢出、NaN/Inf、fully masked 行和恢复；不作 41.5 ms 全额承诺 |
| P1 | **E4 Prefix 全链，再 device KV bridge** | 未插桩 Prefix 45.826 ms；110 copies / 42.06 MB trace | 第一阶段工程目标为降低 Prefix 约 8–15 ms，再测 bridge 增量 | 每层 K/V 不同或内存显著超预算即停止；不能把整个 Prefix 时间视为 overhead |
| P2 | **E5 compact GQA、静态 layout** | Expert layout 8.896 ms，含 repeat / copy / concat；不是全部都能删 | 分离 prefix-static 与 suffix-dynamic，逐项 microbenchmark 后再定收益目标 | `h/3` 映射、cross F32/self BF16 舍入必须一致 |
| P2 | **E6b camera batching / staging 去冗余** | 独立 staging/深拷贝约 6.96 ms；三路串行 graph compute 179.791 ms | 先容量/拷贝复用，再测 batching 的 GPU 增益和内存代价 | 不允许把 batch 改算法的输出差异当作同 backend 逐位等价 |
| 独立轨 | **E7 mixed precision / FA / SDPA** | 本轮四个 GEMM 配置均改动作；此前 FA 严格资格失败 | 仅在明确独立准确率契约后考虑；不更改当前默认 | 不以已有跨 backend 宽容差替代工程优化 bitwise gate |

上述目标区间是按相关 component 规模设置的**实验筛选目标**，不是效果预测、已获得收益或可直接累加的 250 ms 路线。特别是某些 softmax/GEMM 方案很可能无法同时满足现有 bitwise gate，应尽早在 microbenchmark 排除，而不是开发完才发现语义变化。

可用一个明确但未实现的阶段预算检查目标难度：ViT **145 ms**、prepare+other **3 ms**、Prefix **27 ms**、Expert **75 ms**，合计 **250 ms**。相比当前均值，视觉需约 −99 ms、Prefix −19 ms、Expert −35 ms；因此仅靠 Prefix chain、图缓存或 staging 去拷贝不可能完成目标，视觉需要 CPU 预处理与 GPU attention/conversion 同时改善，Expert 也需降低 cast/probe/layout 成本。预算是否可达尚未经实验，不承诺严格路线一定达到 250 ms。

每轮至少做默认/候选/候选/默认交错，先各 31 次，再在收益接近噪声时扩至各 101 次；首轮和第二次 capture/allocator 稳定过程均单列，主报告沿用排除首轮以保留真实缓存启动尾部，另报排除前两次的 steady sensitivity。不用删异常样本来美化 p95。优先门槛为逐位与完整回归通过、端到端 median 改善超过对照漂移且 p95 不回退；<1% 小收益需更多样本证实。设备权限和时钟变化须另行授权，本轮不改。

### 6.6 可复现产物与未完成项

主要产物：`baseline-summary.json`、`baseline-run{0,1,2}.json`、`baseline-run1-clocks.json`、`host-analysis.json`、`trace-analysis{,-stable}.json`、`trace-iteration-stability.json`、`gemm-matrix.json`、`fixture-manifest.json`、两个 `.nsys-rep/.sqlite` capture 及 `scripts/`。模型与逐输入 SHA256、实际 token IDs/valid mask 已写入 fixture manifest；两个 GGUF 哈希与历史记录一致。

复跑脚本均从仓库根执行，临时路径中保存实际绝对命令；重跑 capture 应先使用新的空产物目录或显式归档，避免覆盖报告：

```bash
F=tmp/smolvla-cuda-20261104-1219/comprehensive-20261105
python3 -I "$F/scripts/baseline.py"      # NVTX OFF 主基线
python3 -I "$F/scripts/capture.py"       # CUDA+NVTX+OSRT trace
python3 -I "$F/scripts/analyze.py"       # 排除首轮
PROFILE_EXCLUDE_FIRST=2 python3 -I "$F/scripts/analyze.py"
python3 -I "$F/scripts/capture_host.py"  # NVTX-only 主机细分
python3 -I "$F/scripts/gemm_matrix.py"   # 改数值诊断，非推荐 preset
```

2026-11-05 profiling 轮已实际运行以上所有测量，默认/插桩输出与原 CUDA fixture 逐位一致，**该 profiling 轮没有实施或验收本节新优化**；2026-11-06 的首项实现与验收见下节。尚未完成：本机 LeRobot 同口径 forward、硬件性能计数器、任务级准确率、全机显存峰值/EMC 带宽测量；未重跑历史 Compute Sanitizer 受限项，不声称 memcheck 通过。CPU/default API/HTTP 六组工程回归未在该 profiling 轮重跑，因为当时生产实现未改；这些仍是后续实现的合入门禁。

## 7. 2026-11-06：精确 RGB 坐标复用已实现并验收

### 实现范围

按 P0/E6a 先采用最小的单线程优化，而不是立即增加线程或改变 SIMD/浮点编译参数：

- `src/jetson_pi_smolvla.cpp` 的 `preprocess_rgb()` 每次调用先计算一次横向 x0/x1、fraction 和 byte offset，所有行复用；源/目标行地址移到 x/channel 循环外。
- 保留每个 pixel 的 `/255.0f`、横向→纵向插值、归一化表达式顺序、top/left padding、参数校验、输出/error 清理；无 reciprocal LUT、FMA 选项或 fast-math 变更。
- 坐标表是调用局部 vector，512 宽时在本机约 12 KiB，不缓存输入内容，不引入共享状态、线程池、环境开关或新公共 API；CUDA/CPU 共用并受益。
- 未修改 GPU softmax、FA、图缓存、采样步数和模型权重。

### 严格验证

1. 独立临时 scalar oracle 对照：**1402 组、72,411,684 个 float，逐位零差异**，同时检查返回值和 error。覆盖实际 256→512、identity/downscale、非规则/横竖/极端 aspect、1-pixel、1000 组固定种子随机形状、全部 256 个 byte 常量、null/非法/超限尺寸、陈旧输出清理与失败后恢复。
2. 新增永久 `tests/test_smolvla_preprocess.cpp`，内置冻结的旧 scalar 公式，不下载模型，CUDA 与普通 CPU-only 构建均实际 CTest 通过。同构候选 harness 的 ASan+UBSan 和 `-ffp-contract=off` 对照也通过；这是 CPU 预处理的 sanitizer，不是 CUDA memcheck 通过声明。
3. 先在 tmp 中仅替换 input 静态 archive 链接候选，六组 CUDA policy、policy API、vision reuse 通过后才修改生产源码；候选完整 policy ABBA=401.553→383.489 ms，作为资格证据独立保留，不与下表合并。
4. 应用后实际重建原源码的 CUDA policy/tests、CPU-only policy/tests 和 HTTP server：六组默认 CUDA policy 与原动作逐位一致且 compute CUDA-only，三路 embedding 逐位一致，CUDA/CPU 的 policy API 与 vision reuse 通过；两构建各 6 项 backend/preprocess CTest 全通过。三路 embedding 除隔离候选外，还使用实际重建的 `smolvla-vision-reference` 再次对照保存的旧 embedding，均逐位一致。
5. 实际 HTTP **15 项通过，max_abs=0**，timing 字段及会话错误恢复通过，server 已退出。普通无 CUDA 选项 CPU 构建的完整 policy 与原 CPU 动作逐位一致。

首次 CMake 自动重配置在同一个 multi-target build 中加入新目标时，旧外层 Makefile 尚无该 target rule；保留失败日志，生成后重新执行 build 成功，没有删除测试或修改门禁。

### 最终原仓库构建，未插桩 ABBA

使用保存的旧二进制及共享库作 A、新原仓库构建作 B，顺序 **A→B→B→A**；每组同实例 31 次、剔除首轮，每模式 60 warm，NVTX OFF，无编译/本会话 GPU 实验重叠，未锁频/改权限。所有动作逐位一致。

| 阶段 | 旧路径 mean ms | 新路径 mean ms |
|---|---:|---:|
| 三路 ViT | 244.540 | **226.580** |
| Prefix prepare | 1.961 | 1.965 |
| Prefix prefill | 45.268 | 45.524 |
| 十步 Expert | 109.856 | 109.982 |
| 完整 policy | 402.024 | **384.456** |

- 完整 policy mean 减少 **17.569 ms（4.37%，1.046×）**；median **400.852→382.497 ms**，p95 **405.864→395.273 ms**，min/max **396.525–438.668→379.810–421.631 ms**。Prefix/Expert 未优化，其小波动不归为收益。
- ViT mean 减少 **17.960 ms（7.34%）**；median 244.280→226.236 ms，p95 248.614→228.416 ms。两组旧 mean=244.264/244.815，新=226.896/226.263 ms，改善显著大于前后 control 漂移。
- 排除前两次的 sensitivity：total mean **400.762→383.179 ms**，p95 **405.333→393.123 ms**；主报告仍保留仅排首轮统计，没有删除异常样本。
- 独立 CPU 微基准在另一频率状态下 256→512 median=5.940→3.974 ms、约 −33.1%；不将该单图绝对值外推为实际三相机 policy 节省，也不与前一天 53.967 ms 预处理 trace 直接算加速比。

距 250 ms 目标仍差约 **134.456 ms**，本轮未达到 250 ms。P0 的多线程/SIMD 和 vision softmax 专项仍待测；下一优先项是 1024-column softmax 的严格等价资格实验，以及更进一步的精确预处理优化，不重新推荐已失败的 FA 或全局低精度 GEMM。

产物：`tmp/smolvla-cuda-20261104-1219/preprocess-20261106/`，主要为 `candidate/`、`candidate-validation.json`、`final-validation.json`、`final-ab-summary.json`、四组 `final-ab*.json`、`control/bin/`、`scripts/` 和 `logs/`；HTTP 结果为同会话 `results/preprocess-20261106-http/`。

## 8. 2026-11-06：固定形状 vision softmax 寄存器复用

### 实现及资格实验

- `ggml/src/ggml-cuda/softmax.cu` 只在 NVIDIA CC=870、F32、1024 columns × 1024 rows × 12 heads × batch 1、无 mask/sinks、max_bias=0 时使用新实例；其他设备、形状、类型和带 mask/sinks 的调用保留原 kernel。
- 每线程恰好一个 column，用寄存器保存 scaled score 和 exponential，替代两次 shared-memory 缓存；保持 1024 threads、原循环、`expf`、MAX/SUM 规约树和算术顺序。动态 shared memory 4224→128 bytes；独立编译两实例均 22 registers/thread、无 local memory/spills。没有启用 FA 或改变 BF16/F16/F32 边界。
- 临时直接旧/新 kernel 对照 **12 组 × 12,582,912 = 150,994,944 个值逐位一致**，含 scale=0.125/1/0/−1、随机、极端有限值、signed zero/subnormal、Inf/NaN sign/payload；重复运行也逐位一致。
- 独立 kernel ABBA（100 warmups、15 rounds、每段 50 launches）：scale=0.125 median **2149.605→2099.037 μs**，约 **−2.35%**；保持相同 shared allocation 的对照仍有同等收益。因此这是小幅 kernel 优化，不能将整个约 80 ms 的视觉 softmax trace 时间声称为可消除。
- 永久 `tests/test_smolvla_vision_softmax.cpp` 以零 F32 mask 强制原 fallback 作为有限输入 oracle，同图更新内容，10 次 × 12,582,912 值逐位对照，检查非负/有限及零行 uniform。仅 CUDA 构建注册，无 CUDA device 时 skip=77；该 test 在非 Orin 上可验证通用行为，但不能作为专用 dispatch 命中证明。
- 隔离库先通过三路 embedding、六组完整 CUDA policy（compute CUDA-only）、public policy API 和 vision reuse。隔离 NVTX OFF ABBA 各进程 101 次、每模式 200 warm：total mean **382.299→380.077 ms**、median **381.086→379.038 ms**、p95 **389.908→384.670 ms**；ViT mean **225.971→223.928 ms**。前/后 control total mean=382.204/382.395 ms，两个候选=379.955/380.199 ms；改善方向一致，故进入原仓库实际构建验收，而非依据单 kernel 测试直接合入。

### 原仓库验收

原仓库 CUDA CLI、server、测试及普通 CPU-only 构建成功，最终验收已全部完成：

- 六组默认 CUDA policy 与保存的旧动作逐位一致，compute 节点全 CUDA；三路 embedding 逐位一致，CUDA/CPU policy API 与 vision reuse 通过。CUDA 构建 7 项 backend/preprocess/softmax CTest、CPU-only 6 项 backend/preprocess CTest 全通过。CPU 完整 policy 与原 CPU 基线逐位一致。
- 实际 HTTP 15 项通过、动作 max_abs=0、timing 与错误恢复通过，测试 server 已退出、18087 端口释放。
- 独立 CUDA-node trace 的三次完整 policy 共 **108 次寄存器 softmax**（每次 36），输出逐位一致；这是 dispatch 证据，不把 profiler 时长混入主性能结果。
- 原仓库源码与隔离候选 softmax SHA256 相同；本轮既有源码仅改变 `ggml/src/ggml-cuda/softmax.cu` 和 `tests/CMakeLists.txt`，新增 `tests/test_smolvla_vision_softmax.cpp`。两个 GGUF SHA256 不变，`git diff --check` 通过。未提交/推送、未运行受限的 CUDA memcheck。

### 最终原仓库未插桩 ABBA

A 为本轮开始前保存的实际二进制和共享库（已含上一轮 RGB 优化），B 为原仓库实际重建版本。**A→B→B→A**，每进程 101 次、剔除首轮，每模式 **200 warm**，NVTX OFF；独立于构建、HTTP 回归和 profiler，未锁频/改电源或设备权限。

| 阶段 | 旧 mean ms | 新 mean ms |
|---|---:|---:|
| 三路 ViT | 226.143 | **224.203** |
| Prefix prepare | 1.964 | 1.965 |
| LLM Prefix | 45.076 | 44.173 |
| 十步 Expert | 110.012 | 109.413 |
| 完整 policy | 383.599 | **380.156** |

- 完整 policy mean **−3.443 ms / −0.90%**，median **382.814→379.631 ms**，p95 **387.648→382.659 ms**；min/max **371.634–420.474→374.081–418.030 ms**。所有原始样本保留。
- ViT mean **−1.941 ms / −0.86%**，两个 control mean=226.016/226.271 ms，两个候选=224.383/224.022 ms；total 对应 control=383.544/383.654、候选=380.146/380.166 ms。改善方向在隔离与实际构建中一致，且大于前后 control 漂移。
- 排除前两次 sensitivity：total mean **383.234→379.778 ms**，p95 **387.465→382.342 ms**。Prefix/Expert 没有代码优化；它们约 1.50 ms 的均值变化属于运行波动，**不将完整 3.443 ms 差额全部归因于 softmax**。更可信的局部收益是三路 ViT 约 1.94 ms，隔离 ABBA 约 2.04 ms。
- 接受该小幅优化，但未达到原计划对 softmax 的大幅提速预期；距 250 ms 仍差 **130.156 ms**，不能仅靠这一局部实现达标。不与上一轮的 384.456 ms 跨实验直接计算收益。

下一轮优先独立验证保留中间 BF16 舍入的 F32→BF16→F32 单 kernel、视觉 conversion 复用和 Prefix 全链。此前 FA 与全局低精度 GEMM 的拒绝结论不变，不减少采样步数或删除有限值检查。

产物：`tmp/smolvla-cuda-20261104-1219/softmax-20261106/`，含保存的 `control/bin/`（二进制与共享库）、`candidate/`、`candidate-validation.json`、`candidate-ab-summary.json`、测试和复现脚本。实验不锁频、不修改模型/设备权限、不改变十步采样，不放宽逐位门禁。

## 9. 2026-11-06：保留 BF16 舍入的双 cast 单 kernel 融合

### 实现及安全边界

本轮仅合并 CUDA 执行，不改变 `src/models/smolvla.cpp` 的显式双 cast 图或 finite probe：

- `ggml/src/ggml-cuda/ggml-cuda.cu` 在现有 fusion 入口识别相邻的真实 `ggml_cast`：F32→BF16→F32、同 shape、全部 contiguous、CC=870；中间 BF16 不能是 OUTPUT/view，也不能有其他消费者；目标不能是 view。不适用时继续原双 kernel。
- 实际 `ggml_cast` 是 `GGML_OP_CPY` 且 `src[1]==self`，use count 包含自引用。匹配器使用原 `ggml_node_has_n_uses(...,2)`（自引用 + 唯一下一个 cast）及单节点 `ggml_can_fuse` 检查，而非修改通用 use-count 工具或把所有 CPY 当 cast。
- 直接融合可能缩短源 tensor 的读取生命周期。即使目标没有 view metadata，allocator 仍可能让源/目标部分重叠；因此检查实际地址范围，只允许完全不重叠或精确同址，部分重叠恢复原路径。使用地址差比较避免 endpoint 加法溢出。测试覆盖源/目标相差 ±1 float、精确同址和恰好无重叠。
- `cpy.cu` 的新 kernel 使用与原 copy 完全相同的 `ggml_cuda_cast<float>(ggml_cuda_cast<nv_bfloat16>(x))`，保留 BF16 中间舍入，沿用 64-thread copy block、launch/PDL hook。不是 identity，也不是主机 bit-mask；尤其 CUDA 原 NaN canonicalization 不能替换成不同的 CPU conversion。
- 不改变图 allocator/public API，不承诺减少实际 graph tensor buffer 内存；中间 BF16 图节点仍存在，只省 launch 与 device 中间读写。无新开关，沿用已有全局 fusion-disable 行为，但不拿“关闭所有 fusion”的路径冒充只关闭本优化的严格性能对照。

### 隔离严格资格与延迟门禁

1. 独立 CUDA 直接旧 copy kernels 两次 vs 新 kernel：**42,665,412 个输入 / 50 batches / bitwise 零差异**。含 8,388,608 随机 F32 bits、全部 BF16 编码×7 舍入邻域、全部 F32 NaN payload/sign 与 infinities（16,777,216）、全部 F32 subnormal/sign 与 signed zero（16,777,216）、极值/溢出邻域和 tail 长度。比较旧路径输出 bits，不声称原输入 NaN payload 被保留。
2. 新增图级 test 的 oracle 将两次 cast 分别放在两个 compute graph，中间 BF16 主机读回/再上传，确保 oracle 不能被新 matcher 融合；另保留一个 OUTPUT-pinned BF16 cast 链并直接检查中间输出。覆盖多消费者、后续 view alias、转置/非连续输入、奇数/tail、真实 720×50/2048×50/960×241、内容反复更新、精确同址/±1 float 部分重叠和不重叠。**28,908,800 个 F32 输出值逐位一致**，中间 BF16 OUTPUT 同时通过。
3. 在 tmp 中仅替换 CUDA 的 `cpy.cu.o` 和 `ggml-cuda.cu.o`，其他对象来自已验收版本。带 overlap guard 的最终候选三路 embedding、六组完整 CUDA policy（compute 全 CUDA）、policy API 和 vision reuse 通过；未修改原模型/采样/精度。
4. 初版隔离 ABBA total mean 381.263→366.054 ms，仅作为探索记录。发现 partial-overlap 生命周期边界后增加 guard，完整测试与 ABBA 重跑；不会将初版无 guard 的测量作为最终候选验收。
5. **带 guard 候选 NVTX OFF ABBA**：A→B→B→A，每进程 101 次、剔首轮、每模式 200 warm；total mean **380.860→365.953 ms**、median **379.518→365.331 ms**、p95 **383.671→368.074 ms**；Prefix mean **44.576→39.792 ms**，十步 Expert **109.876→99.667 ms**，ViT **224.030→224.111 ms**。前/后 control total=379.835/381.884，两候选=366.247/365.658 ms；主收益大于 control 漂移。排除前两次 sensitivity total=380.278→365.537 ms。所有动作逐位一致。

### 原仓库最终验收

**已通过最终验收并默认启用**，无额外构建/运行时选择变量：

- 实际重建原仓库 CUDA CLI/vision/tests/server 和普通无 CUDA 选项 CPU-only 目标；CUDA **8/8** 项 backend/preprocess/softmax/BF16-round CTest、CPU **6/6** 项 backend/preprocess CTest 全通过。
- 六组默认 CUDA policy 与保存的旧 CUDA 动作逐位一致，所有 compute 节点为 CUDA；三路 embedding 逐位一致，CPU 完整 policy 与旧 CPU 基线逐位一致，CPU/CUDA policy API 与 vision reuse 通过。
- 真实 HTTP **15 项通过、max_abs=0**，timing 与错误恢复通过；server 已退出，18087 端口已释放。
- 独立三次完整 policy 的 CUDA-node trace：新融合 kernel **9222 次（3074/次 policy）**，未融合的 F32→BF16 与 BF16→F32 各 **75 次（25/次 policy）**，动作逐位一致。这是 dispatch 证据；不同 trace 轮的总时长不混入未插桩性能，也不据此声称实测 DRAM 带宽或 graph buffer 节省。
- 实际生产三文件与带 guard 的隔离候选逐一 SHA256 相同，两个 GGUF SHA256 不变，`git diff --check` 通过。该轮既有源码只改 `ggml-cuda.cu`、`cpy.cu`、`cpy.cuh`、`tests/CMakeLists.txt`，新增 `tests/test_smolvla_bf16_round.cpp`；未修改模型图、probe、CPU 或普通非 CC870 CUDA 路径。未提交/推送，未运行此前受限的 CUDA memcheck。

### 最终原仓库未插桩 ABBA

A 为本轮开始前保存的二进制与匹配动态库（已含 RGB/softmax 优化），B 为原仓库实际重建版本。**A→B→B→A**，每进程 101 次、剔除首轮，每模式 **200 warm**，NVTX OFF，与构建/其他 GPU 实验不重叠，未锁频或改变电源/权限；全部动作逐位一致。

| 阶段 | 旧 mean ms | 新 mean ms |
|---|---:|---:|
| 三路 ViT | 224.299 | 224.672 |
| Prefix prepare | 1.981 | 2.000 |
| LLM Prefix | 44.790 | **40.730** |
| 十步 Expert | 110.089 | **100.354** |
| 完整 policy | 381.560 | **368.157** |

- 完整 policy mean **−13.403 ms / −3.51%**，median **380.354→367.217 ms**，p95 **390.928→374.398 ms**，min/max **369.685–420.484→358.392–417.824 ms**。
- Prefix mean **−4.060 ms / −9.07%**，Expert **−9.735 ms / −8.84%**；ViT/prepare 不属于此优化范围，其小幅变化不作收益归因。
- 两组 control total mean **382.392/380.727 ms**，两组新版本 **368.230/368.084 ms**；即使存在约 1.67 ms 的 control 漂移，收益仍显著更大，两组方向一致。候选轮与最终轮不混合统计，不把最终新均值比候选高解释成代码回退。
- 排除前两次 sensitivity：total mean **381.178→367.727 ms**，p95 **389.985→372.921 ms**，改善方向保留；主报告仍采用只排首轮，保留全部原始首轮/异常样本。
- 距 250 ms 仍差 **118.157 ms**，尚未达到目标。不与上一轮另一环境状态的 380.156 ms 跨实验算收益。当前 ViT 约占 total **61%**，接下来应优先量化视觉 conversion/operand 复用和进一步严格等价 softmax 布局实现，再推进 Prefix 全链；finite probe 融合必须保留失败语义，不能用删除检查或减少采样步数追赶目标。

产物：`tmp/smolvla-cuda-20261104-1219/bf16-round-20261106/`，含保存的旧 binary/libs、初版与带 guard 的 patch/ABBA、独立数值日志、图级测试、模型回归与脚本。不锁频、不改电源/权限、不启用失败的 FA、不放宽门禁。

## 10. 2026-11-06：视觉 Q/K/V 输入转换复用

**状态：未采纳，已回退。** 隔离资格通过后曾应用并实际重建；数值/功能回归通过，但原仓库最终 ABBA 的全链路 p95 回退，性能门禁失败。下面候选实现均仅保存在 tmp，不是生产默认实现，不把局部或隔离轮改善记作已交付收益。

### 实现与精度边界

- `clip_graph::build_mm` 仍保持普通 `ggml_mul_mat`。只对固定 strict CUDA Idefics3 的 separate Q/K/V 分支创建一次共享 F16 输入，原 F32 normalized `cur` 不变，bias、residual、attention、FFN、projector 均不变。
- 限定设备 description 精确 `Orin`、输入 contiguous F32 `[768,1024,1,1]`、三组 contiguous F16 `[768,768,1,1]` 权重及既有 512×512/batch1/FA-disabled reuse 资格。1024 列原调用不满足 MMVF/MMF 的小列数门限，本就进入 cuBLAS；显式 F16 输入不会把该 shape 从原 custom kernel 切到另一套 GEMM。
- graph CPY 与 cuBLAS 内部 `convert_unary` 都使用同一 `ggml_cuda_cast<half>`。保留 F16 输入舍入及原 F16 GEMM/临时输出/转回 F32；不采用之前严格数值失败的全局 F32-output、BF16 compute 或 FA 方案。
- 显式 `GGML_CUDA_CUBLAS_COMPUTE_TYPE`（包括 auto/空串/未知值）或 `GGML_CUDA_LT_AUTOTUNE` 均禁用共享，不重复实现 CUDA 的配置解析。资格在每次 encode 前检查，切换时 invalidate 常驻 CLIP 图；CPU/其他架构/普通 CLIP/其他 shape/融合 QKV 分支保持原图。
- 额外修复 capture 的 runtime precision 生命周期：旧 CUDA Graph 只比较 UID/tensor metadata，没有包含每次 GEMM 读取的 compute override。即使重建 CLIP 图，相同 metadata 仍可能重放旧精度。候选在每个 CUDA capture 状态中保存 override 的完整值与是否设置，变化时在 UID 快路径之前失效，重新 warmup/capture。F16-output 和 Lt 等原本进程内锁存的配置不改为动态语义；不支持并发无同步地修改进程环境。
- 新增 shared F16 graph tensor 跨三次投影存活，不宣称减少 graph allocator 峰值内存。逻辑上少两次 F32→F16 转换，不等于实测 DRAM 带宽或 PCIe 流量。

### 数值资格与被拒绝的中间实现

1. 独立 ggml GEMM：三组投影、768 宽、1023/1024/1025 列，随机/舍入邻域/subnormal/±0/Inf/NaN/极值及反复内容更新；old F32 RHS 对比 shared F16 RHS，**35,389,440** 个输出逐位一致。
2. 默认隔离三路 embedding、六组完整 policy（compute 全 CUDA）、policy API/vision reuse 通过。初版共享实现默认路径虽正确，但额外 runtime 精度切换测试失败，因此没有直接应用。
3. 两个长期保留 handle 的最初 oracle 也会重放旧精度；改为每种设置新开一个原 rebuild-path handle。旧 control 的同实例 unset→f32 切换失败，证实是既有 capture 问题。仅在 CLIP 保存完整配置 key 仍不足以修复 f16→auto，必须失效后端 capture。
4. 修正后同一实际 vision handle 的 **12 组场景 / 24 次 embedding** 与新建原路径 oracle 逐位一致，含 absent/F32/BF16/F16/auto/空/未知值、显式 Lt=0 与恢复默认。另一个无模型 capture test 以新建 CUDA backend 作 oracle，**65,536** 个输出逐位一致；旧 control 在 f32 repeat0 失败，新实现通过。
5. 全部 mtmd 40 个 TU 与 CUDA 139 个 TU 在隔离 overlay 一致重编译，避免修改 `clip_graph`/capture 状态布局却混用旧类定义。生产源码在上述资格及扩展 ABBA 之前保持不变。

### 独立性能测量：保留首轮 p95 回退

各轮均 A→B→B→A，binary 与 `LD_LIBRARY_PATH` 成对，固定模型/输入/noise、NVTX OFF，不锁频、不与构建/其他 GPU 工作重叠，动作逐位一致。各轮独立统计，不将不同环境状态的绝对均值相减计算收益。

- 初版 sharing-only 探索轮（每模式 200 warm）：ViT mean **224.594→221.865 ms**，total **366.256→363.955 ms**、p95 **376.203→369.052 ms**；该实现不满足完整精度切换资格，不作为最终接受版本。
- 完整修正首轮（每模式 200 warm）：ViT **223.603→221.096 ms**，total **364.478→362.077 ms**，median **364.375→361.299 ms**，但 p95 **368.033→370.092 ms（回退 2.059 ms）**。排除前两次 p95 仍 **367.974→368.528 ms**，因此当时不通过性能验收，没有只挑均值采纳。
- 为评估尾延迟不确定性扩大独立 ABBA，每进程 **201 次**、每模式 **400 warm**：ViT mean **224.107→221.490 ms**、p95 **225.095→222.111 ms**；total mean **365.260→362.073 ms**、median **364.839→361.697 ms**、p95 **368.861→364.464 ms**。两个 control total=365.862/364.658，两个候选=362.465/361.680 ms；ViT control=224.134/224.080、候选=221.369/221.610 ms。排除前两次 total mean **365.088→361.859 ms**、p95 **368.692→364.213 ms**。首轮回退仍保留，扩展轮不与首轮混合。
- 当前隔离可信局部收益约三路 ViT **2.62 ms**；不是 250 ms 达标结果。原仓库实际构建与最终 ABBA 尚须单独确认。

### 独立 dispatch trace

三次完整 policy（含 cold）的 trace 与保存 control 比较：cuBLAS 内部 F32→F16 **4026→3702** 次，新增 shared F32→F16 CPY **108** 次，F16→F32 **4026→4026** 次，GEMM kernel 名称/计数 **5301→5301** 且逐项相同。每 policy 净省 **72** 个 conversion kernel，保留原 GEMM 路径。trace 结果逐位一致；只作 dispatch 证据，不能用 trace 总时长替代独立 ABBA，也不报告为实测带宽。

### 原仓库最终验收

实际 CUDA/CPU/server 构建成功；候选两项 CUDA 单测与既有回归（CUDA **10/10**、CPU **6/6** CTest）、三路 embedding/六组 CUDA-only policy、CPU 完整 policy、同实例环境切换、两 backend API/vision reuse、HTTP **15 项 / max_abs=0** 均通过，独立 production trace 完成。source/model 哈希和端口检查也执行至最终性能断言之前通过。但这些不替代尾延迟门禁：

| 阶段 | 旧 mean ms | 被拒绝候选 mean ms |
|---|---:|---:|
| 三路 ViT | 224.428448 | 221.867935 |
| Prefix prepare | 1.945788 | 1.999498 |
| LLM Prefix | 39.205460 | 39.616310 |
| 十步 Expert | 99.343428 | 99.453055 |
| 完整 policy | 365.326773 | 363.339548 |

- 实际构建独立 ABBA，每进程 201 次、剔首轮、每模式 **400 warm**，NVTX OFF，动作逐位一致。总 mean **−1.987 ms / −0.54%**，median **364.866→362.400 ms**，但总 **p95 366.987→369.455 ms（+2.468 ms）**。两个 control total mean=365.593/365.060 ms，候选=363.690/362.989 ms；候选两组 p95=368.340/374.963 ms，不能只报告更好的均值。
- 排除前两次 total mean **365.072→363.085 ms**、p95 **366.981→369.415 ms**，回退方向仍在。ViT mean **−2.561 ms / −1.14%**、p95 **224.828→222.423 ms**；Prefix/Expert 未优化，其运行波动不能证明转换共享本身导致了尾延迟，也不能作为忽略全链路回退的理由。当前未锁频环境下，本轮未获得可重复的全链路尾延迟改善。
- `finalize.py` 在 `candidate.total.p95 <= control.total.p95` 断言失败、退出 1；**没有生成 accepted execution-summary，也不追加挑选通过的性能轮次**。此前扩展隔离轮的通过不能覆盖原仓库最终失败，候选不采纳。
- 已按原源码快照和受限反向 patch 恢复四个实现文件及 CMake，删除仅本轮新增的两项测试；逐文件 SHA256 与本轮起点一致。实际被拒绝 binary/libs 保存到 `production-candidate/bin/` 后，main CUDA binary/libs 与 `control/bin` 逐字节恢复；server 恢复先前已验收构建，CPU 按恢复源码正常重建。CUDA 对象缓存仍由正常依赖时间戳在下次构建重编译，不将隔离/候选对象冒充恢复源码的构建结果。
- 恢复后的三路 embedding、六组 CUDA-only policy、CPU 完整 policy、两 backend API/vision、CUDA 8/8 与 CPU 6/6 CTest、HTTP 15 项 / max_abs=0 全部通过；模型 SHA256 不变、18087 端口释放、diff check 通过。最终生产保持此前已验收的 RGB、softmax 寄存器和 BF16 round-trip 优化；本轮发现的旧 capture precision 配置切换问题仍存在于生产，只保留已验证的隔离修正供以后独立处理，不声称本轮已交付该修复。
- 本轮不提供新的“已验收总延迟”或 250 ms 达标声明，前节 BF16 已验收数据仍有效。下一实验优先保持原 32 列 warp 分组与两级规约顺序的 vision softmax 多元素/线程布局；仅为后续设计，尚未实现，不借此假定收益。

产物：`tmp/smolvla-cuda-20261104-1219/vision-convert-20261106/`，含保存的原 binary/libs 和五个原源码快照、初版/修正/扩展独立 ABBA、失败及成功回归日志、完整隔离源码/对象/编译链接命令。未下载/安装、未改硬件配置或模型、未减少十步采样、未提交/推送。
