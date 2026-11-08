# SmolVLA CUDA 优化方案与执行记录

## 目标与基线

目标是减少当前 staged SmolVLA 路径中的重复计算、主机往返及细粒度同步，不改变默认十步采样、50×32 内部动作、三摄像头输入或显式 BF16 舍入语义。当前默认 backend 跟随构建：`GGML_CUDA=ON` 默认 CUDA，CPU-only 默认 CPU；CUDA 不静默回退。

2026-11-04 Orin、CUDA arch 87、Release、4 CPU threads，固定输入，同实例 7 轮、剔除首轮的已有基线：

| 阶段 | CUDA warm mean ms | 占比 |
|---|---:|---:|
| 三路 ViT | 261.867 | 17.1% |
| Prefix prepare | 2.135 | 0.14% |
| LLM prefix prefill | 209.913 | 13.7% |
| 十步 Action Expert | 1058.970 | 69.1% |
| 完整 policy | 1533.278 | 100% |

这些是包含主机处理及传输的阶段 wall time，不是 kernel-only 时间。现有 CUDA Graph replay 已启用；优化目标是扩大常驻计算范围，而非重复开启已有功能。旧 trace 含加载及三次推理，无 NVTX 阶段归因，不能把同步 API 累计时长当作全部可消除开销。

## 分阶段方案

### A. 不变量提升与无用计算裁剪（先执行）

1. 每次采样开始预计算八个 cross-attention 层的 prefix K/V 投影，由 160 次投影 GEMM 降至 16 次；保持投影结果 F32。缓存限定当前请求，避免同 shape 不同内容误命中。
2. 每请求构造一次 suffix mask 与 self/cross positions，复用十步；复用固定 shape 工作区。Q 和 self suffix K/V 仍每步变化，不能缓存其值。
3. Prefix 最后一层只计算输出 K/V 所需的 norm、K/V projection、K unpermute/RoPE/舍入；不计算无消费者的 Q、attention、residual 和 MLP。保留全部 16 层 K/V。
4. 为相同输入的旧/新路径比较保留可控基线开关或独立旧二进制，测量前固定输入和噪声。

### B. Device-resident 图改造（A 验证后推进）

1. 连接 Expert 单层 QKV、layout、RoPE、attention、residual/MLP，移除层内 host 向量中转，保留所有 BF16 rounding 节点。
2. 再连接完整单步图，prefix K/V 和 hidden 常驻 device；最后再考虑 GPU Euler 与十步图。
3. GQA 必须保留 h/3 头映射；self 与 cross 的舍入、位置和 mask 差异不能合并抹平。全 mask 行当前使用有限 -FLT_MAX，不能改成 -infinity。
4. 每个子阶段分别验证与测量；不能删除保护 backend 共享 scratch 的锁来冒充优化。异步 API 本身无法消除主机消费者引起的等待。

### C. 转换与 kernel 优化（B 后按实测选择）

- 等价 F32→BF16→F32 单 kernel 或保留中间舍入的 elementwise fusion。
- 避免 CPU 上三倍展开 KV heads，验证 native GQA 布局。
- 独立进程 A/B 测试 GGML_CUDA_LT_AUTOTUNE 与 GGML_CUDA_F16_F32_OUT；不能引用其他模型的收益作为 SmolVLA 实测。
- 不直接删除 casts 或强制 FP16。新 kernel/归约顺序需要独立数值资格验证。

### D. ViT 与部署（独立后续阶段）

- 先细分预处理、graph setup、GPU compute、readback，再做固定图与 staging buffer 复用。
- 评估 ViT Flash Attention；三摄像头 batching 需修正 Idefics3 pixel-shuffle 的单图假设，不能只翻开关或把摄像头拼成同一 attention 序列。
- 清理 server 通用路径与专用 policy 重复加载，主要收益是内存与冷启动。
- 更少采样步数、量化、降低分辨率属于模型/数值改变，不纳入等价工程优化。

## B1 后的调优优先级（2026-11-04）

以下是下一阶段计划，不代表已实现。按 B1 未插桩 warm mean，ViT / Prefix / Expert 分别约占完整 policy 的 39% / 32% / 28%，主要瓶颈已不再只有 Expert。

### 现有 trace 的依据

重新分析已有 `optimization/nvtx-resident.sqlite` 的第 2、3 次推理，未重新采集或据此更新未插桩延迟：

| 范围（每次推理平均） | kernel 数 | kernel 累计 ms | GraphLaunch | StreamSynchronize | memcpy activity 数 / bytes |
|---|---:|---:|---:|---:|---:|
| Prefix | 1034 | 23.310 | 77 | 353 | 276 / 176880504 |
| Expert | 15228 | 117.360 | 198 | 844 | 646 / 74005760 |

- Prefix 仍有 16 个 QKV、31 个 RoPE、15 个 attention、15 个 residual/MLP compute 边界；已存在缓存，问题不是每次都缺少 CUDA Graph。CPU unpermute、舍入、GQA 展开和 hidden 中转是下一步目标。
- 每路 ViT 有 419 个 kernel、约 61.4 ms kernel 累计，其中 softmax 约 26.7 ms；每路仅一个 GraphLaunch。warm 范围未见 cudaMalloc/cudaFree，不能把图重建等同于每次 GPU 分配。
- Expert 的 5620 个 BF16 往返 cast kernel 累计约 28.667 ms；finite probe 的 scale/reduce 累计约 10.647 ms（不含 flag 合并）；repeat 为 320 个、约 4.567 ms。这些只说明值得研究，不是可直接相加的端到端收益。
- tensor set/get 内部有 stream synchronize；同步时间包含有效 GPU 工作等待。Orin memcpy bytes 不是 PCIe 流量，trace wall time 减 kernel 累计也不是全部可消除开销。

### 执行顺序与验收

1. **P0：Prefix 单层常驻图，再连接层间 hidden。** 先在 CUDA 连接 QKV、unpermute、NEOX RoPE、相邻三头 GQA、attention、残差/MLP；CPU 与 staged 对照保留。保持层 0/后续层 norm 舍入差异、所有显式 BF16 往返、有限 `-FLT_MAX` 全 mask 行、全部 16 层 canonical K/V，以及末层 K/V-only 裁剪。先验证每层 K/V 逐位一致，再验证完整动作、失败/复用行为，最后决定是否默认开启。
2. **P1：完整 Expert velocity-step 常驻。** 先连接 16 层 hidden，理论上每样本减少 300 次 hidden 边界传输、43.2 MB；代码 compute 路径目标由 198 降至 48，再合并 encode/norm/velocity 至 18（含 8 次 cross-KV 预计算）。这些是路径规划，不是实测 GraphLaunch 或延迟承诺。GPU Euler 保留到后续，避免改变 FMA/舍入。
3. **P1（独立资格验证）：ViT Flash Attention。** 当前通用 FA 会把 K/V 转 F16；必须独立比较三路 embedding、全部 Prefix K/V、动作与失败行为，不能作为严格等价替换直接启用。失败则保留关闭，记录差异，不放宽现有门禁。
4. **P2：BF16 往返单 kernel、finite-check fusion、固定 vision 图/staging 与预处理流水线。** 保留原舍入点与非有限值检查语义；不能删除验证来提速。复用应覆盖 shape/content 变化和失败恢复，不能复用旧请求的值。
5. **P3：native compact GQA、真实三摄像头 batching、GPU Euler/十步图、独立 GEMM tuning。** batching 先解决 Idefics3 pixel-shuffle 的单图假设；Prefix embedding device bridge 保持 image `sqrt(960)` 缩放、language 缩放后 BF16 舍入、state 不缩放的不同语义。GEMM 环境开关按独立进程验证数值，不引用其他模型收益。

本次新增执行产物放在 `tmp/smolvla-cuda-20261104-1219/post-b1/`，其中 `baseline/bin` 已保存修改前实际默认 CUDA 二进制和共享库。每个阶段分别记录实现状态、失败诊断、严格数值比较、未插桩交错 A/B；未完成项继续标记待办。正常构建契约保持不变：全新无 CUDA 参数为 CPU，CUDA ON / GRAPHS ON 为默认 CUDA，无需 SmolVLA 环境变量。

## 验证与测量门禁

- 临时输入、基线备份、构建、日志及 trace 放在 tmp/smolvla-cuda-20261104-1219/ 下的优化子目录；不修改模型、电源、时钟或设备权限，不操作机器人。
- 从原仓库构建，先保存优化前实际二进制及必要动态库/输出，避免构建覆盖后失去比较基准。
- CPU/CUDA 固定六组完整输入，对比优化前同 backend 输出；不改变数学路径的修改以逐位一致为目标。若存在差异，先定位，不能直接放宽阈值。
- 保留 CPU/CUDA graph reuse、A→B→A、不同内容同 shape、失败行为、timing 开关、HTTP 检查。
- 同实例 7 次，首轮单列；顺序测量，记录 warm mean/median/min/max。前后环境不锁频，需警惕热/功耗波动，必要时交错复测。
- profiling 若加入 NVTX，应覆盖推理编号、camera/prefix/step/layer，并将 instrumentation 与未插桩端到端测量分开。不得新增全局 cudaDeviceSynchronize 作为性能计时手段。
- 已知限制继续保留：独立组件跨 backend 合成测试 29/53 超差；无本次独立 CUDA LeRobot/PyTorch oracle；设备 memcheck 因 GPU debugging disabled 未完成；无机器人任务验证。

## 执行与复现

可选 NVTX 标记默认关闭。沿用现有构建配置，加 `-DSMOLVLA_NVTX=ON -DCUDAToolkit_ROOT=/usr/local/cuda` 启用；需 CUDA toolkit 中的 `nvtx3/nvToolsExt.h`。使用 header-only NVTX v3，不依赖旧 nvToolsExt 动态库。范围包含推理序号、三路 camera、prefix prepare/prefill、expert，以及已实现的 Expert step/layer、cross-KV precompute 和 resident layer。未插桩性能测量使用 `-DSMOLVLA_NVTX=OFF`。

模型级复用测试不下载模型：

```bash
SMOLVLA_BACKEND=cuda "$B/bin/test-smolvla-policy-reuse" \
  ../models/smolvla_gguf/f16/smolvla-main.gguf \
  ../models/smolvla_gguf/f16/mmproj-smolvla.gguf
```

CPU 使用 `SMOLVLA_BACKEND=cpu`。无模型参数的 CTest 会显式 skip（77），不是模型推理通过。

CUDA Expert 常驻路径默认启用，并在本轮升级为 16 层链；`SMOLVLA_EXPERT_CHAIN=0` 恢复 B1 单层常驻，`SMOLVLA_EXPERT_RESIDENT=0` 使用 A 的 staged 对照路径，`1` 显式允许常驻路径。CPU 和 public standalone forward 始终保持 staged，可作对照。此开关不改变十步采样、精度或动作维度；已通过下面记录的正常路径模型级验收，不能据此宣称所有模型/输入均已验证。只设置 `GGML_CUDA=ON`、`GGML_CUDA_GRAPHS=ON` 即可编译并默认使用 CUDA，不再要求 backend/resident 环境变量。下面实验记录中的 opt-in/default 描述保留当时状态，不代表当前默认设置。

```bash
SMOLVLA_BACKEND=cuda SMOLVLA_EXPERT_CHAIN=0 SMOLVLA_EXPERT_RESIDENT=1 \
  "$B/bin/test-smolvla-sample-reuse" ../models/smolvla_gguf/f16/smolvla-main.gguf

# NVTX ON 构建时，另行采集，不将此运行用作无插桩延迟结果：
SMOLVLA_BACKEND=cuda SMOLVLA_EXPERT_CHAIN=0 SMOLVLA_EXPERT_RESIDENT=1 SMOLVLA_BENCH_REPEATS=3 \
  nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx \
  --cuda-graph-trace=node --output=/path/to/new-output \
  "$B/bin/smolvla-policy-reference" MODEL MMPROJ IMAGE0 IMAGE1 IMAGE2 STATE NOISE TEXT OUTPUT
```

上述 B1 单层对照仍保留层间 hidden 主机读回；当前默认层链已消除该部分，但 encode、最终 norm/velocity 和 Euler 边界仍独立，不是完整单步/十步 GPU 常驻。为保持组合图边界生命周期而固定部分 device tensor，增加缓存内存，尚未最小化；不能把它宣传为零拷贝或零额外内存方案。

## 执行记录

- A 已实现并从原仓库构建成功（NVTX OFF）。Cross prefix K/V 投影由每样本 160 次降为 16 次；mask/positions 从每层构造改为每请求一次；复用主机工作区；裁掉 prefix 最后一层无消费者的 Q/attention/residual/MLP。
- A 的 14 项实际回归通过：CPU/CUDA graph reuse、六组完整输入同 backend 与优化前逐位一致、sample 与未缓存手工 Euler 逐位一致、图像/指令/状态/有效 mask A→B→A 与 fresh B、噪声复用、HTTP 动作/阶段计时、非法 backend/device。详见本会话 `optimization/phase-a-validation.json`。
- 已保存优化前可执行文件和动态库至本会话 `optimization/baseline/bin`；运行基线副本时显式设置该目录的 `LD_LIBRARY_PATH`，避免旧二进制错误链接后续新库。
- 新增模型级 A→B→A、fresh B、失败恢复测试，涵盖图像/指令/状态/有效 mask 内容变化；已构建并在 CPU/CUDA 实际运行通过。
- B1 已实现并构建：CUDA opt-in 单 Expert 层常驻图，保留 staged standalone 路径作为独立对照。首次 sample-vs-uncached-Euler 逐位比较失败（合成 case 0，max_abs=3.21621084，1600/1600 元素不同）。失败日志保存在 `optimization/logs/resident-sample-first.log` 与 `resident-sample-diagnostic.log`；不放宽阈值、不启用默认。
- B1 诊断进展：临时保留并逐项比较 projected Q/K/V、RoPE、GQA 展开、attention、hidden；将边界 tensor 及 view allocation root 保守标为 OUTPUT 后，8169 项逐层 tensor 对照和 sample 测试通过。INPUT 本身不保证静态 buffer 跨 compute 保留；静态 prefix/mask/positions 也需固定生命周期。诊断代码已从源码删除，保留必要 device 边界保护。随后无诊断的正常路径 sample 测试与完整 14 项 CPU/CUDA 回归全部通过，六组完整 policy 与优化前同 backend 输出逐位一致（`optimization/resident-validation.json`）；并非仅依据 debug 结果验收。尚未隔离最小必要 pin 集合，不将此定位泛化为已证明的通用 ggml allocator 缺陷。
- 全部 16 层 prefix K/V 的额外旧/新对照通过：CPU/CUDA 均在 1→3→2→3 tokens、含全 mask query 行的输入上逐位一致（每 backend 比较 368640 bytes），见 `optimization/prefix-comparison.json`。
- B/C/D 依赖前阶段结果；只有实际完成构建、回归与测量的项目才标记完成，其余保持待办，不把假设收益写成实测结果。

### A 实测结果

同一设备未锁频；优化前保存的旧二进制/库与 A 的保存二进制/库分别顺序运行，每组同实例 7 次、剔除首轮。这是一次前后测量，不能据此泛化到其他输入/设备。原始 JSON 为 `optimization/before-{cpu,cuda}.json` 和 `optimization/phase-a-{cpu,cuda}.json`。

| 阶段 | 优化前 CUDA warm mean ms | A 后 CUDA warm mean ms |
|---|---:|---:|
| 三路 ViT | 260.046 | 259.938 |
| Prefix prepare | 2.121 | 2.106 |
| LLM prefill | 213.477 | 202.851 |
| 十步 Expert | 1063.925 | 976.852 |
| 完整 policy | 1539.966 | 1442.142 |

A 的完整 CUDA policy 本轮减少约 97.824 ms（6.35%，1.068×）；Expert 减少约 87.073 ms（8.18%）。CPU 完整 policy 为 8828.286→8717.631 ms。优化前后 benchmark 动作与保存的同 backend 原始输出逐位一致。注意不要将本表与较早一次 1533.278 ms 基线混作同一前后实验。

稍后再次运行原始保存二进制，CUDA total=1554.988 ms、Expert=1075.092 ms（`optimization/control-cuda.json`），相对首次约 1% 波动，说明无锁频环境下应保留控制组。A 的扩展 sample 测试在 CPU/CUDA 上通过非法 mask、noise shape/Inf、KV shape/NaN、极大有限 cross prefix、失败恢复及输入/输出 alias 检查。

### B1 正常路径实测结果

同一最终二进制，NVTX OFF，分别设置 `SMOLVLA_EXPERT_RESIDENT=1` 与 `0` 顺序运行。每组 7 次、剔除首轮，以下是剩余 6 次 warm mean；原始逐次数据及 median/min/max 位于 `optimization/resident-cuda.json`、`optimization/final-staged-cuda.json`。无锁频，ViT 等未改阶段亦有波动，不将全部端到端差额归因于 Expert kernel。

| 阶段 | 最终 staged mean ms | B1 resident mean ms |
|---|---:|---:|
| 三路 ViT | 262.828 | 248.557 |
| Prefix prepare | 2.133 | 2.004 |
| LLM prefill | 214.105 | 205.583 |
| 十步 Expert | 1008.708 | 181.526 |
| 完整 policy | 1488.184 | 638.071 |

- 相对同二进制 staged：完整 policy 约 **2.33×**，延迟减少 **57.12%**；Expert 约 **5.56×**，延迟减少 **82.00%**。
- 相对本轮最初保存原始二进制的 1539.966 ms，完整 policy 约 **2.41×**，延迟减少 **58.57%**。不要与更早一次 profiling 表混为同一实验。
- B1 total median=626.195 ms，min/max=617.937/670.157 ms；Expert median=178.858 ms，min/max=173.148/194.115 ms。首轮 total=886.614 ms，单独保留，不算入 warm 平均。
- 两种模式 benchmark 动作均与优化前 CUDA 输出逐位一致。B1 的 CPU/CUDA 14 项回归包括 HTTP 实际请求、不同内容/有效 mask/shape、fresh B、失败恢复、非有限/极大有限输入及 alias；CPU 忽略 resident 开关，仍走 staged。
- 静态源码计数：Expert 每样本的同步 graph compute 从 A 的 758 次降至 B1 的 198 次（包含 8 次 cross-KV 预计算及 30 次 encode/norm/velocity）；这是代码路径计数，不等于 CUDA kernel 或 cudaGraphLaunch 次数。
- 保守固定的 16 层边界 tensor 逻辑大小约为 `13,824,000 + 122,880 × prefix_tokens` bytes，241 tokens 时约 41.43 MiB；不是实测额外显存峰值，未计静态输入、对齐、其他缓冲区或 allocator 复用差异。

### NVTX 实测归因

已实际配置并编译 `SMOLVLA_NVTX=ON`，分别采集 staged/resident 各 3 次完整推理，输出动作与优化前 CUDA 基线逐位一致。原始文件为本会话 `optimization/nvtx-{staged,resident}.nsys-rep`、同名 SQLite 和 `nvtx-expert-comparison.json`。范围覆盖推理 ID、camera、prefix、10 steps、16 layers；resident 另有 resident-layer 范围。

下面仅统计各自第 2、3 次 **Expert NVTX 范围内**的事件，每次取平均，不包含模型加载、ViT、prefix 或首轮；runtime API 另限定同一主机线程，GPU 活动按完整时间区间归属：

| 每次 warm Expert | staged | B1 resident |
|---|---:|---:|
| cudaGraphLaunch | 758 | 198 |
| cudaStreamSynchronize | 3340 | 844 |
| cudaMemcpyAsync / memcpy activity | 2582 | 646 |
| memcpy activity 总 bytes | 626540160 | 74005760 |
| kernel activity 数量 | 10748 | 15228 |

这验证了减少 graph 边界与主机往返的机制，但没有消除所有 host crossing。kernel 数增加与设备 GQA 展开、有限值 probe 等有关，不能以 kernel 数量单独判断性能；有限值检查没有为了提速而删除。Orin 物理内存统一，上表 bytes 不应称作 PCIe 流量。插桩 trace 的时间不替代前述 NVTX OFF 延迟；同步 API 时长包含等待有效 GPU 工作，不能全部作为可消除的开销。

### B1 后的新一轮执行（Prefix 单层完成，ViT FA 拒绝）

- 已将上面的 P0–P3 优先级和 trace 依据纳入本方案；P1/P2/P3 不因列入计划而视为完成。
- 已保存当前默认 CUDA 二进制及共享库到 `post-b1/baseline/bin`，并记录 SHA256 manifest。
- 新一轮修改前实际运行固定输入、同实例 7 次、剔除首轮，NVTX OFF、未设置 backend/resident 变量：ViT=247.941 ms，prepare=2.021 ms，Prefix=199.267 ms，Expert=183.008 ms，total=632.642 ms（warm mean）。动作与原始 CUDA 基线逐位一致，逐次数据和统计见 `post-b1/baseline-current.json`。这是新控制组，不替换历史 B1 表。
- Prefix 单层常驻候选已通过 16 层 canonical K/V staged/resident 逐位对照：1→3→3（内容改变）→2（全 mask query）→3，以及现实长度 1→241→241（内容改变）→2→241；CPU 仍 staged。额外失败门禁覆盖 FLT_MAX、1e30、mask=2、Inf：两路径接受/拒绝一致，拒绝时 K/V 清空，随后有效请求逐位恢复。FLT_MAX 在两路径均拒绝（报错描述可不同），1e30 均接受且 K/V 逐位一致。
- **ViT FA 已验证并拒绝作为等价优化启用。** 三路 embedding 均不逐位一致（每路 61440 元素中约 59000 个不同，max_abs=0.21875 / 0.21875 / 0.25）；六组完整 policy 各有 299–300 / 300 动作不同，max_abs 约 0.0078125–0.0370245。FA OFF 与原 CUDA 输出逐位一致；A→B→A 复用检查通过不代表精度等价。因已在 embedding/动作门禁失败，未继续为 FA 捕获全部 Prefix K/V，未进行收益验收。临时诊断开关已从源码移除，默认继续关闭；诊断补丁及详细数据保存在 `post-b1/vision-fa/diagnostic.patch`、`qualification.json`，还原构建后动作复核逐位一致。未放宽数值门禁。
- `post-b1/validation.json` 的 20 项实际调用全部通过：六组完整输入各跑 Prefix staged/resident（12 项，均与原 CUDA 输出逐位一致且 policy compute 全为 CUDA）、CPU/CUDA 各三项 Prefix/sample/policy reuse、graph reuse、backend CTest（含 5 个选择测试）。保留手工 Euler、失败恢复及 timed/untimed 检查。
- 严格门禁通过后，CUDA Prefix 单层常驻已默认开启；无需新增构建参数或环境变量。`SMOLVLA_PREFIX_RESIDENT=0` 独立恢复 Prefix staged，`SMOLVLA_EXPERT_RESIDENT=0` 独立恢复 Expert staged；CPU 始终 staged。最终重新构建后，无任何 `SMOLVLA_*` 变量的六组 CLI 与 public policy API reuse 再次通过（`final-default-validation.json`）。真实 HTTP 15 项检查通过，动作对直接 CUDA 参考 max_abs=0，阶段计时有效，测试 server 已退出；HTTP 报告在会话 `results/post-b1-http/report.json`。

#### Prefix 常驻最终未插桩 A/B

同一最终二进制、NVTX OFF，固定输入、4 threads。按 Prefix `0→1→1→0` 顺序运行四组，每组同实例 7 次，分别排除首轮；每种模式合计 12 个 warm 样本。Expert 保持默认 resident，FA 关闭。设备未锁频，以下是本次交错运行结果，不是跨设备保证。全部输出与原 CUDA 动作逐位一致。

| 阶段 | Prefix staged mean ms | Prefix resident mean ms |
|---|---:|---:|
| 三路 ViT | 248.297 | 248.993 |
| Prefix prepare | 2.012 | 2.039 |
| LLM prefix prefill | 201.108 | 49.694 |
| 十步 Expert | 178.958 | 173.772 |
| 完整 policy | 630.782 | 474.904 |

- Prefix 约 **4.05×**，阶段延迟减少 **75.29%**；完整 policy 约 **1.33×**，减少 **24.71%**（155.878 ms）。未改的 Expert 也有约 5 ms 波动，不将所有端到端差额算成 Prefix kernel 收益。
- Prefix staged/resident median 为 197.341 / 44.375 ms，min/max 为 195.836–219.850 / 43.128–77.396 ms；完整 policy median 为 625.182 / 467.764 ms，min/max 为 623.580–660.130 / 463.353–511.574 ms。
- 原始首轮、逐次 warm 与统计见 `post-b1/final-ab{0,1,2,3}-prefix{0,1}.json`（实际文件按组对应模式），汇总为 `final-ab-summary.json`，脚本为 `compare_bench.py`。早期探索性短测不作为收益依据。
- Prefix 同步 compute 代码路径由 77 个子图合为 16 个层图；仍每层上传 hidden/positions/mask、读回 hidden/全部 canonical K/V 和检查标量，并非层链常驻或零拷贝。边界 tensor/view root 保守固定有额外缓存开销；本轮未测独立 allocator 显存峰值，未宣称零额外内存。

### Expert 层链执行（已通过数值与延迟门禁）

- 目标：把十步采样中的每步 16 个 Expert 层接成一个设备图，先不合并 encode/norm/velocity，不改 host Euler、步数、动作维度或精度。理论代码 compute 路径为 198→48，移除 300 次层间 hidden 上传/读回、43.2 MB；这些是结构目标，不是延迟预测或实测 trace 数。
- 保留 public staged forward / 手工 Euler 和单层 resident 对照，候选最初只用 `SMOLVLA_EXPERT_CHAIN=1` 启用。必须保持每层 hidden/原中间检查、BF16 舍入、layer0 norm 差异、self/cross softmax 精度差异、相邻三头 GQA 和全 mask 有限最小值语义。静态 K/V、mask、positions 每个请求刷新；形状变化和失败恢复不能使用旧值。
- 同一 handle 在 chain `0→1→0` 切换的回归已加入 `test_smolvla_sample_reuse.cpp`，对照手工 Euler；沿用 NaN/Inf/FLT_MAX、mask/shape 错误、恢复和 alias 门禁。六组完整动作须与原 CUDA 基线逐位一致，不能放宽门禁。
- 新产物为 `tmp/smolvla-cuda-20261104-1219/expert-chain/`，已保存实际修改前二进制/库和 SHA256 manifest。新控制组 NVTX OFF、同实例 7 次、排除首轮：ViT=248.883、Prefix=49.919、Expert=174.898、total=476.140 ms（warm mean），动作逐位一致。此值单独记录，不与上一轮不同实验混算收益。
- `expert-chain/validation.json` 的 **21 项实际调用全部通过**：六组完整输入各跑 chain=0/1，全部与原 CUDA 输出逐位一致且 policy compute 全为 CUDA；显式 resident=0/chain=1 仍选 staged 并输出一致；CPU/CUDA Prefix/sample/policy reuse、graph reuse、5 项 backend CTest 全通过。扩展 sample 门禁保留手工 Euler、同 handle 模式切换及原失败/alias 检查。
- 数值与下述交错性能门禁通过后，CUDA Expert **默认启用 16 层常驻链**，不需额外参数/环境变量。`SMOLVLA_EXPERT_CHAIN=0` 恢复单层常驻，`SMOLVLA_EXPERT_RESIDENT=0` 恢复完全 staged，CPU 始终 staged。最终重建后，六组无任何 `SMOLVLA_*` 环境变量的 CLI / public policy API reuse 再次通过，并检查日志确实分配 chain（`final-default-validation.json`）。
- 真实 HTTP 15 项检查通过，动作对直接 CUDA 参考 max_abs=0、阶段计时有效，未设置 backend/device/resident/chain 选择变量，server 日志确认默认 chain，测试 server 已退出（会话 `results/expert-chain-http/report.json`）。最终普通无 CUDA 参数构建重新编译并实际运行完整 CPU policy、sample/manual Euler、5 项 backend CTest 全通过，动作保持原 CPU 逐位一致（`final-cpu-no-options.json`、`logs/cpu-only-sample.log`）。

#### Expert 层链最终未插桩 A/B

同一最终图实现、NVTX OFF、固定输入、4 threads；chain `0→1→1→0` 四组，每组同实例 7 次并分别排除首轮，每模式 12 个 warm 样本；Prefix 保持默认 resident、FA 关闭。该测量在默认 predicate 切换前完成，切换后未改计算图。所有动作逐位一致，设备未锁频。

| 阶段 | 单层 Expert resident mean ms | 16 层链 mean ms |
|---|---:|---:|
| 三路 ViT | 247.230 | 250.361 |
| Prefix prepare | 1.996 | 1.927 |
| LLM prefix prefill | 49.514 | 49.288 |
| 十步 Expert | 170.846 | 118.941 |
| 完整 policy | 469.983 | 420.930 |

- Expert 约 **1.44×**，延迟下降 **30.38%**；完整 policy 约 **1.12×**，下降 **10.44%**（49.053 ms）。未改阶段也有波动，不将所有端到端差额直接等同于 Expert kernel 收益。
- 单层/层链 Expert median=171.297 / 118.511 ms，min/max=162.659–176.079 / 117.972–121.100 ms；total median=464.760 / 413.768 ms，min/max=450.072–515.355 / 408.016–462.676 ms。
- 原始首轮、warm 样本与统计为 `expert-chain/final-ab{0,1,2,3}-chain{0,1}.json`（对应实际模式），汇总 `final-ab-summary.json`，脚本 `compare_bench.py`。不与历史 474.904 ms 的另一轮实验混算本轮收益。
- **实测 scheduler CUDA buffer**：241 tokens，16 个单层图合计 95,418,368 bytes（91.00 MiB），层链 65,543,680 bytes（62.51 MiB）；详见 `scheduler-buffers.json`。这是计算 scheduler buffer，不含权重、其他图、CUDA Graph/driver 分配或进程/device 峰值。模式切换测试会同时保留两套缓存，不能称为整体显存节省证明。
- 独立 CUDA-only trace、NVTX OFF、两模式各三次完整推理（含首轮），实际 cudaGraphLaunch 总数 **611→176**，两模式动作逐位一致（`graphs-chain{0,1}.json`、同名 `.nsys-rep/.sqlite`）。此总数包含冷启动，不直接当作每次 warm/Expert 阶段计数，也不以 trace 时长替代未插桩延迟。
- 本轮只连接 Expert 层链。下一步可合并 encode/final norm/velocity，之后再研究 finite probe / BF16 casts 融合和 ViT 固定图/staging；GPU Euler/十步图与 FA 仍未启用。

### 完整 Expert velocity-step 执行（已通过数值与延迟门禁）

- 将 action/time encode、16 层 Expert、最终 norm 和 velocity projection 接成单步图；保持 CPU Euler/十步采样和全部显式舍入。时间正弦 embedding 若涉及 CPU/GPU trig 差异，保留原 CPU 计算后上传，不能以近似换取等价验收。
- 最初仅 `SMOLVLA_EXPERT_STEP=1` 选择候选，保留 step=0 的已验收层链；显式 chain=0 / resident=0 继续选各自旧对照。必须保留 encode/层间/final norm/velocity 的非有限值门禁、shape/content 变化、失败清零/恢复和 alias 行为。
- 目标为每样本同步 compute 代码路径 48→18（8 个 cross-KV 预计算加 10 个完整 step）；这是结构规划，不是实测延迟或 launch 保证。需要实测 scheduler CUDA buffer size，不能预设融合一定减少内存或端到端延迟。
- 新产物放在 `tmp/smolvla-cuda-20261104-1219/expert-step/`；已保存修改前实际默认 CUDA 二进制/库及 SHA256 manifest。固定输入、NVTX OFF、同实例 7 轮剔除首轮的本轮控制组：ViT=249.912、Prefix=49.144、Expert=119.365、total=420.816 ms，动作与原 CUDA 参考逐位一致。与历史另一轮的 420.930 ms 区分记录。
- 扩展永久 sample 测试，在同句柄选择单层、层链、完整 step、返回层链，每种结果对照 staged public forward / 手工 Euler。数值与未插桩交错 A/B 均通过后才决定默认启用，CPU/CUDA 默认构建契约不变。

#### 完整 step 数值与未插桩 A/B

- `expert-step/validation.json` 的 **22 项实际调用通过**：六组完整 policy 各跑 step=0/1，均与原 CUDA 参考逐位一致且 compute 全为 CUDA；resident=0、chain=0 均优先覆盖 step=1；CPU/CUDA Prefix/sample/policy reuse、graph reuse、backend CTest 均通过。同句柄单层→层链→完整 step→层链对照 public staged forward / 手工 Euler 逐位一致，保留失败清零、恢复、alias 与形状/内容/mask 变化检查。额外 FLT_MAX/1e30 noise 对照同样拒绝并恢复，不放宽阈值。
- 保留原 CPU double pow/sin/cos 后转 float 的时间 embedding；CPU Euler、十步数和所有 BF16 舍入不变。encode/各层 hidden/final norm/velocity 非有限值检查以设备侧标量 probe 保留，不把这些检查从融合路径删除。
- 同一最终图实现、NVTX OFF，step `0→1→1→0`，每组同实例 7 次、分别排除首轮，每模式 12 个 warm 样本；固定输入、4 threads、未锁频，动作全部逐位一致。测量在默认 predicate 修改前完成，之后未更改计算图。

| 阶段 | 16 层链 mean ms | 完整 velocity-step mean ms |
|---|---:|---:|
| 三路 ViT | 250.475 | 247.564 |
| Prefix prepare | 1.995 | 1.977 |
| LLM prefix prefill | 50.048 | 53.379 |
| 十步 Expert | 120.412 | 112.098 |
| 完整 policy | 423.333 | 415.406 |

- Expert 延迟下降 **6.90%**（8.313 ms，1.074×），完整 policy 下降 **1.87%**（7.927 ms，1.019×）。Expert median=119.224 / 109.597 ms、min/max=118.204–131.036 / 109.183–128.460 ms；total median=415.088 / 405.896 ms、min/max=413.758–461.651 / 402.936–462.354 ms。Prefix 与 ViT 也有明显波动，不能将全部端到端差额归因于 kernel 改进，不与历史另一轮 420.930 ms 混算收益。
- 原始首轮/warm 数据为 `expert-step/final-ab{0,1,2,3}-step{0,1}.json`（按实际模式），汇总为 `final-ab-summary.json`，复现脚本为 `compare_bench.py`。
- 数值和性能门禁通过后，完整 step 已设为 CUDA 默认路径，不需额外变量。`SMOLVLA_EXPERT_STEP=0` 选层链，`SMOLVLA_EXPERT_CHAIN=0` 优先选单层，`SMOLVLA_EXPERT_RESIDENT=0` 优先选 staged；CPU 继续 staged。最终重建后六组无 `SMOLVLA_*` 的 CLI / public policy API reuse 全通过，日志确认默认 step 且 CUDA compute 无 CPU fallback（`final-default-validation.json`）。
- 真实 foreground HTTP **15 项检查通过**，动作对直接 CUDA 参考 max_abs=0、阶段计时有效，日志确认默认 step；无 backend/device/resident/chain/step 选择变量，测试 server 已退出（`results/expert-step-http/report.json`）。HTTP 首次推理含冷启动，不并入 warm A/B。最终无 CUDA 配置参数的 CPU-only 构建也完成真实完整 policy（与原 CPU 逐位一致）、sample/manual Euler 和 5 项 backend CTest（`final-cpu-no-options.json`）。
- 实测 241-token Expert scheduler CUDA buffer：层链 **65,543,680 bytes**，完整 step **65,550,080 bytes**，增加 6,400 bytes，约 62.51 MiB；不声称整体显存下降。该数不含权重、其他图、driver/CUDA Graph 分配或进程峰值，模式切换还可能同时持有不同缓存。
- 独立 CUDA-only trace、NVTX OFF，每模式三次完整推理，实际 `cudaGraphLaunch` 总数 **176→89**，两模式输出逐位一致（`graphs-step{0,1}.json` 和同名 `.nsys-rep/.sqlite`）。包含冷/暖迭代，不直接等同于每步或每次 warm 数量，不使用 trace 时长代替未插桩性能。

### ViT 已分配图复用（已通过数值与延迟门禁；未改 staging）

- 优先排查三路视觉调用中重复构图、scheduler allocation、输入/输出 staging；保持原始逐算子公式、FA 关闭，不缓存图像内容或跳过任一摄像头。
- 产物为 `tmp/smolvla-cuda-20261104-1219/vision-reuse/`，已保存完整 step 默认路径实际二进制/库及 manifest。独立控制组 NVTX OFF，同实例 7 次剔除首轮：ViT=251.941、Prefix=49.535、Expert=109.960、total=413.865 ms；动作与原 CUDA 参考逐位一致。此控制组不与历史另一轮直接混算加速比。
- 接受条件：三路 embedding 和六组完整 policy 对旧路径逐位一致；同句柄 A→B→A、输入尺寸变化、失败恢复与 CPU 路径不受影响；未插桩交错 A/B 显示收益后才默认启用。通用 CLIP 路径应保持原默认行为；缓存须限定适用模型/形状并明确失效与销毁顺序。
- 实现只在严格 CUDA、FA disabled、单图非 audio、512×512、无特殊 image flags 的 Idefics3 上保留一个已分配 ggml 图。持有 metadata context，不持有引用调用方 image 的 builder；非命中、模式切换或失败时先同步/reset scheduler 再释放 metadata，任何共享 arena 的重新构图入口均先失效。销毁 scheduler/buffer 先于 backend。仍每次上传新像素、完整计算，不缓存 embedding。
- `vision-reuse/validation.json` **24 项实际调用通过**：三路 embedding 各跑 reuse=0/1 对原 FA-OFF CUDA embedding 逐位一致；六组完整 policy 各跑 0/1 对原 CUDA 动作逐位一致且 compute 全 CUDA；CPU/CUDA vision/policy reuse、backend CTest、graph reuse 通过。新增 `test_smolvla_vision_reuse.cpp` 用独立 staged 句柄作 oracle，同一候选句柄在内容、256×256/320×192/1×1 原始宽高、0/1 模式间变化并返回 A，覆盖非法宽度后清零/恢复；预处理后的图形状始终为 512×512，不把原始宽高变化称作多图形状缓存验证。

#### ViT 图复用未插桩 A/B

固定输入、4 threads、NVTX OFF、未锁频，reuse `0→1→1→0`，每组同实例 7 次排除首轮，每模式 12 个 warm 样本；Expert 保持已验收完整 step。测量后只修改默认 predicate，未改计算图或内核。

| 阶段 | 逐图重建 mean ms | 已分配图复用 mean ms |
|---|---:|---:|
| 三路 ViT | 250.314 | 245.841 |
| Prefix prepare | 1.907 | 1.981 |
| LLM prefix prefill | 50.164 | 51.386 |
| 十步 Expert | 110.577 | 110.602 |
| 完整 policy | 413.363 | 410.217 |

- ViT 下降 **1.79%**（4.473 ms，1.018×），完整 policy 下降 **0.76%**（3.146 ms，1.008×）；收益较小，不能泛化为任意输入/部署的稳定收益。ViT median=249.010 / 244.237 ms、min/max=247.796–255.539 / 243.596–251.150 ms；total median=405.519 / 400.611 ms、min/max=400.134–454.232 / 398.100–461.663 ms。Prefix 波动较大，不能把全部端到端差额归为 vision kernel 收益。
- 两组旧路径 ViT mean=249.650 / 250.978 ms，两组新路径=245.265 / 246.417 ms；两组都显示相同改善方向。每组 7 次共 21 路视觉调用，旧路径严格 allocation 检查 21 次，新路径 1 次、缓存创建 1 次；这些是日志生命周期证据，不等于 cudaMalloc 数量或 GPU 总内存节省。
- 原始首轮、warm 与统计为 `vision-reuse/final-ab{0,1,2,3}-vision{0,1}.json`（对应实际模式），汇总 `final-ab-summary.json`；所有动作逐位一致。数值/性能门禁通过后默认开启，`SMOLVLA_VISION_REUSE=0` 恢复逐图重建；CPU/普通非 strict CLIP 路径不启用。
- 最终默认验证通过：六组无 `SMOLVLA_*` 的 CLI 输出逐位一致、public policy API 和强化后的双句柄 vision reuse 测试通过；真实 HTTP 15 项通过、动作 max_abs=0，日志确认仅创建一次缓存，server 已退出。首次 HTTP 日志断言因 common logger 将 ggml INFO 映射到 trace verbosity=4 而未显示标记；保留失败记录，提高测试日志级别后全部重跑通过，未删除断言或修改计算。普通无 CUDA 选项 CPU-only 构建完成真实 policy（原 CPU 逐位一致）、vision reuse 和 5 项 backend CTest（`final-default-validation.json`、`final-cpu-no-options.json`、`results/vision-reuse-http/report.json`）。
- 独立 CUDA-only trace，每模式三次完整推理，`cudaGraphLaunch` 总数均为 **89**，动作逐位一致（`graphs-vision{0,1}.json` 及 `.nsys-rep/.sqlite`）。GPU launch 数不变符合只消除主机构图的范围；不把 graph cache 命中解释为跳过 GPU 计算，不使用 trace 时长替代未插桩延迟。
- 本轮未改 CPU RGB preprocess、HWC→CHW staging 分配/拷贝、camera batching、FA 或模型计算；输入 staging 容量复用和冗余拷贝移除仍是独立待测项。

### 2026-11-05：转向算子与预处理优化，而非继续仅优化构图

详见 [Orin 深度 profiling 与 250 ms 目标](smolvla-orin-latency-analysis-20261105.md)，包含原始证据、LeRobot 比较口径、具体代码入口、实验预算与拒绝条件。本轮没有修改生产实现或接受新优化。

独立 NVTX OFF 60 个 warm 的 total mean/median/p95 为 **401.915 / 401.017 / 419.345 ms**；ViT=244.015、Prefix=45.826、Expert=109.695 ms。NVTX-only 确认三路预处理约 **53.967 ms**；展开 CUDA Graph 的 steady trace 确认视觉 softmax **80.066 ms**、视觉 GEMM 转换 **28.248 ms**、Prefix/Expert BF16 cast **9.821 / 20.851 ms**。逐节点 tracing 显著增加 wall，不能用 trace 数字直接预测未插桩收益；它们用于排序和约束成本上限。

更新后的实验顺序：

1. **P0：精确 CPU resize/normalize 优化 + 1024-column vision softmax 专项**。预处理行并行、坐标预计算、SIMD 必须保留原运算/舍入；softmax 线程或 reduction 改序也必须先过逐元素门禁，不能默认等价。
2. **P1：视觉 GEMM operand/output conversion 共享和 layout/bias 融合**，以及 **BF16 round/finite probe 精确融合**。先量化共享对象，保留全部 BF16 舍入与有限大值溢出/NaN/Inf 检查，不将 `0*x` 优化为常量零。
3. **P1：Prefix 全链，再内部 device KV bridge**。保留末层仅 K/V、公共 host API 和 CPU 行为；逐层对照 K/V，独立测 bridge 增量，不把 110 次拷贝全部解释成纯传输耗时。
4. **P2：compact GQA / static KV layout、视觉 staging 和三相机 batching**。保持 `h/3` 分组、self/cross 精度差别、每请求刷新和 batch-aware pixel unshuffle；不缓存输入内容。
5. **独立精度轨：mixed precision / FA / SDPA**。本轮 F16/F32-out、加 LT autotune、强制 BF16/F16 均改变 300/300 动作；强制类型更慢，全部不采用。此前 FA 失败记录不变，不能作为严格路线的默认优化。

当前到 250 ms 的均值缺口为 **151.915 ms**（约需 −37.8%、1.608×）；仅优化 staging、Prefix chain 或已有图缓存不够。设计阶段预算 ViT145 + prepare/other3 + Prefix27 + Expert75 =250 ms，**这是尚未实现的目标，不是加速预测**；不得将相互重叠的理论收益相加声称已经可达。每项完成后继续六组严格回归、同句柄异常恢复、CPU/API/HTTP 和未插桩交错 A/B，延迟小于噪声时扩大样本，保留 p95。设备频率未锁定、CPU DVFS 差异、LeRobot 尚无本机同口径运行、硬件计数器权限受限均在完整报告中单列。

### 2026-11-06：P0 精确 RGB 坐标复用（已实现、验证并应用）

- 首项采用无需新增线程/API/开关的单线程实现：`preprocess_rgb()` 调用局部复用横向 x0/x1/fraction/offset，预计算源/目标行地址；保留 `/255.0f`、横向后纵向插值与 padding/normalize 顺序。512 宽坐标表本机约 12 KiB，不缓存像素，无共享状态；CPU/CUDA 都使用该路径。
- 在 tmp 先进行旧 scalar oracle 对照：1402 组、72,411,684 float 逐位一致，覆盖所有 byte、随机/非规则/极端尺寸、非法输入与恢复；input-only ASan+UBSan 和禁用 FMA contraction 的独立对照通过。新增永久 `test_smolvla_preprocess.cpp`，不依赖模型。没有改变编译精度参数。
- 隔离候选完整 policy 六组/API/vision 与未插桩 ABBA 通过后应用到原仓库。实际重建 CUDA、普通 CPU-only 和 server；六组默认 CUDA 动作、三路 embedding 逐位一致，CUDA/CPU policy API/vision reuse 和各 6 项 backend/preprocess CTest 通过，CPU 完整 policy 与原 CPU 基线逐位一致，HTTP 15 项 max_abs=0，server 已退出。
- 最终原仓库 NVTX OFF **A→B→B→A**，每进程 31 次剔首轮、每模式 60 warm：ViT mean **244.540→226.580 ms（−7.34%）**，total mean **402.024→384.456 ms（−17.569 ms，−4.37%）**；total median **400.852→382.497 ms**，p95 **405.864→395.273 ms**。两组对照和两组候选均显示视觉改善；Prefix/Expert 未优化，其小波动不归为收益。
- 排除前两次 sensitivity total mean **400.762→383.179 ms**；所有原始样本保留。未锁频、未改模型/设备权限；不将另一 CPU 频率状态下约 33% 的单图 microbenchmark 降幅外推为端到端收益。
- 距 250 ms 尚差约 **134.456 ms**，未完成目标。下一步继续 1024-column vision softmax 严格等价资格实验；多线程/SIMD、conversion/round/probe、Prefix chain 等仍待独立验证，不放宽 bitwise 门禁。详见 [深度分析执行记录](smolvla-orin-latency-analysis-20261105.md)。

产物：`tmp/smolvla-cuda-20261104-1219/preprocess-20261106/`，包括 `final-ab-summary.json`、`final-validation.json`、独立候选与保存的 control binary/libs。此前 profiler-only 记录和历史 A/B 原样保留。

### 2026-11-06：P0 固定形状 vision softmax 寄存器复用

- 最小实现限定 NVIDIA CC 8.7、F32、1024×1024×12×1、无 mask/sinks、max_bias=0；每线程一个值保留在寄存器，原 1024 线程与 MAX/SUM reduction tree 不变，其他调用走原 fallback。shared scratch 4224→128 bytes，无新开关，不启用 FA、不变更模型精度。
- 直接 kernel 12 组 / 150,994,944 个值（含 NaN payload）逐位一致；独立 kernel median 2149.605→2099.037 μs，仅约 −2.35%。永久 CUDA test 用零 F32 mask 原 fallback 作有限输入 oracle，不复制 kernel；覆盖同图内容更新、零行 uniform、极值及 signed zero，10 组逐位对照。
- 隔离库三路 embedding、六组 CUDA policy、API/vision 通过。小收益按计划扩大至每进程 101 次，ABBA 每模式 200 warm：total mean 382.299→380.077 ms、p95 389.908→384.670 ms，ViT mean 225.971→223.928 ms；两个 control 与两个 candidate 分别同向。只有通过原仓库实际构建后的完整回归与 ABBA，才记录为最终验收；不将跨轮 DVFS 差异归为收益。
- **最终原仓库验收通过，已默认应用**：实际 CUDA/CPU-only/server 构建、六组 CUDA 动作/三路 embedding 逐位一致、CPU policy 逐位一致、两 backend 的 API/vision reuse、CUDA 7 项及 CPU 6 项 CTest、HTTP 15 项（max_abs=0）通过。三次 policy trace 命中 108 次新 softmax，动作逐位一致。GGUF 哈希不变，server/端口已释放，diff check 通过。
- 实际构建 NVTX OFF ABBA，每进程 101 次、每模式 200 warm：ViT mean **226.143→224.203 ms（−1.941 ms / −0.86%）**；total mean **383.599→380.156 ms（−3.443 ms / −0.90%）**、median **382.814→379.631 ms**、p95 **387.648→382.659 ms**。两组 control/candidate 均同向；排除前两次 total mean=383.234→379.778 ms。Prefix/Expert 未优化，约 1.50 ms 的变化不归于 softmax；局部可信收益约 2 ms。距 250 ms 仍差 **130.156 ms**，未达目标，不与上一轮跨实验算加速比。最终结果见 [深度报告第 8 节](smolvla-orin-latency-analysis-20261105.md#8-2026-11-06固定形状-vision-softmax-寄存器复用)。产物：`tmp/smolvla-cuda-20261104-1219/softmax-20261106/`，含 `final-validation.json`、`final-ab-summary.json`、`trace-summary.json` 和 `execution-summary.json`。
- 该实验说明：保持严格逐位语义的局部 softmax 改善远小于整个 softmax 成本。下一轮优先独立验证 **F32→BF16→F32 单 kernel**（保留中间 BF16 舍入、NaN/Inf/溢出行为）与 conversion 复用，随后 Prefix 全链；finite probe 融合仍须保留 `0*x` 和原失败边界，不通过删除检查或重开已失败 FA 追求 250 ms。
- 下一实验入口：`src/models/smolvla.cpp` 的双 `ggml_cast` round helper，以及 `ggml/src/ggml-cuda/cpy.cu` 的 F32→BF16 / BF16→F32 dispatch。应先验证单消费者、布局/stride、view/output 生命周期与 alias，再考虑 backend fusion；不能仅删除中间 tensor，因为中间 BF16 是模型舍入语义，不是无用 cast。先用独立旧 cast 链比较全部特殊值与随机 bit patterns，再跑完整模型/API/失败恢复和相同 ABBA。此项目前仅完成代码入口定位，**尚未实现或声称收益**。

### 2026-11-06：P1 精确 BF16 round-trip 单 kernel

- 采用 backend fusion，保留模型图中的 F32→BF16→F32 舍入点及 finite probe。只匹配 Orin CC=870、相邻真实 cast、同 shape contiguous、BF16 单外部消费者且非 OUTPUT/view；`ggml_cast` 的自引用计入 use count，不修改通用融合工具。只允许 source/destination 完全不重叠或精确同址，部分重叠回退，避免 allocator 生命周期缩短导致数据竞争。
- CUDA 原 conversion helpers 直接组合，不用 CPU conversion/bit-mask 替代。42,665,412 输入 bits（含全部 NaN/subnormal 编码）旧双 kernel vs 新 kernel 逐位一致。图级两独立 compute graph 作强制旧 oracle，28,908,800 值逐位一致，覆盖 OUTPUT 中间值、多消费者/view、转置/非连续、tail、内容更新、±1 float 部分重叠和精确 in-place。
- 初版资格结果保留，但在发现 partial-overlap 边界后未直接应用，带 guard 的最终候选完整重跑。三路 embedding、六组 CUDA policy（compute 全 CUDA）、API/vision reuse 逐位通过。
- 带 guard 候选 NVTX OFF ABBA，每进程 101 次、每模式 200 warm：total mean **380.860→365.953 ms**，median **379.518→365.331 ms**，p95 **383.671→368.074 ms**；Prefix **44.576→39.792 ms**，Expert **109.876→99.667 ms**，ViT 未改善。数值/候选性能门禁通过后应用到原仓库，**最终实际构建与回归/性能门禁均通过，已默认启用**；最终生产构建验收见 [深度报告第 9 节](smolvla-orin-latency-analysis-20261105.md#9-2026-11-06保留-bf16-舍入的双-cast-单-kernel-融合)。
- 最终原仓库 NVTX OFF ABBA，每进程 101 次、每模式 200 warm：Prefix mean **44.790→40.730 ms（−9.07%）**，Expert **110.089→100.354 ms（−8.84%）**，total **381.560→368.157 ms（−13.403 ms / −3.51%）**；median **380.354→367.217 ms**，p95 **390.928→374.398 ms**。ViT 224.299→224.672 ms 未优化；两组 control 与新版本均同向，排除前两次 total mean=381.178→367.727 ms。距 250 ms 仍差 **118.157 ms**，尚未达标，不跨轮混算收益。
- 实际 CUDA/CPU-only/server 构建、六组 CUDA policy/三路 embedding/CPU policy 逐位一致、两 backend API/vision reuse、CUDA 8/8 与 CPU 6/6 CTest、HTTP 15 项（max_abs=0）通过。三次 policy trace 新 kernel=**9222**，输出逐位一致，仅作 dispatch 证据；GGUF 哈希不变，server/端口释放，diff check 通过。下一优先项为视觉 conversion/operand 复用，随后 Prefix 全链及保留失败语义的 probe 融合。
- 产物：`tmp/smolvla-cuda-20261104-1219/bf16-round-20261106/`，初版 `candidate-ab*`、带 guard 的 `guarded-ab*` 和实际构建 `final-ab*` 分开保留，最终为 `final-validation.json`、`final-ab-summary.json`、`trace-summary.json`、`execution-summary.json`。无新 API/变量、不改时钟/电源/设备权限、不减少十步数。没有减少图中 buffer allocation 的保证；优化仅针对 kernel 与中间读写。

### 2026-11-06：P0 视觉 Q/K/V 输入转换复用

- 仅固定 strict CUDA Idefics3 / Orin，contiguous F32 `[768,1024,1,1]` 输入与三组 F16 `[768,768,1,1]` 权重；原 1024-token 调用已走 cuBLAS，排除 custom MMVF/MMF，使用与内部转换相同的 CUDA scalar cast。只共享 separate Q/K/V 输入，不替换 residual、不修改 attention/FFN/projector 或 GEMM 输出/累加精度。
- 显式 `GGML_CUDA_CUBLAS_COMPUTE_TYPE` 或 `GGML_CUDA_LT_AUTOTUNE` 禁用共享；每 encode 检查资格，变化时丢弃 CLIP 常驻图。额外发现 CUDA capture 未记录 runtime compute override，单纯重建 CLIP 图仍会重放旧精度；已在候选的 capture 状态中加入覆盖值/是否设置，检查置于 UID 快路径之前。原 control 的独立回归能复现 f32 切换失败。
- 直接 GEMM **35,389,440** 个输出逐位一致；三路 embedding、六组完整 CUDA-only policy、API/vision reuse、同一 handle 12 组环境切换/24 次 embedding 通过；capture 独立 fresh-backend oracle **65,536** 值通过。共享分支初版虽通过默认模型回归，未通过 runtime 精度切换，未直接采纳。
- 修正候选首轮 ABBA 每模式 200 warm：ViT mean **223.603→221.096 ms**，total **364.478→362.077 ms**，但 total p95 **368.033→370.092 ms 回退**。据此扩展独立 ABBA 至每进程 201 次、每模式 400 warm：ViT **224.107→221.490 ms**，total **365.260→362.073 ms**，median **364.839→361.697 ms**，p95 **368.861→364.464 ms**；两轮不合并、不隐藏首轮尾延迟回退。排除前两次 total mean=365.088→361.859 ms，p95=368.692→364.213 ms。
- 三次 policy 的独立 trace：内部 F32→F16 4026→3702，新增共享 CPY 108 次，F16→F32 4026 不变，GEMM kernel 名称及计数 5301 不变；每 policy 净省 72 个转换 kernel。仅作 dispatch 证据，不报告为实测 DRAM 带宽或图内存节省。
- **最终原仓库性能门禁失败，未采纳，已回退本轮四个实现文件、CMake 注册及新增两项测试。** 实际 CUDA/CPU/server 构建及数值/API/vision/HTTP 全通过，CUDA 10/10、CPU 6/6 CTest；但独立 400 warm/mode ABBA：ViT mean **224.428→221.868 ms**，total **365.327→363.340 ms**，median **364.866→362.400 ms**，总 **p95 366.987→369.455 ms（回退 +2.468 ms）**。排除前两次 p95 仍 **366.981→369.415 ms**，不能只以 mean 改善认定接受，也不再反复测到某轮通过。
- 原源码逐文件 SHA256 已恢复，CUDA main-build binary/libs 逐字节恢复保存的 control，server 恢复先前已验收版本、CPU 按恢复源码重建；恢复后完整回归通过（CUDA 8/8、CPU 6/6 CTest，三路 embedding/六组 CUDA policy、CPU policy、两 backend API/vision、HTTP 15 项 / max_abs=0）。原生产实现仍是上节已验收的 BF16 round-trip 版本；本轮 runtime capture 配置切换修正同样不留在生产，已知旧问题未宣称解决。仅文档记录本轮新结论。产物含 `final-ab-summary.json`、`final-validation.json`、`production-candidate/bin/`、原源码快照、失败/成功日志和 rollback 脚本：`tmp/smolvla-cuda-20261104-1219/vision-convert-20261106/`。
- 下一轮优先**保留原 32 列 warp 分组及两级规约顺序的 vision softmax 多元素/线程布局**，再考虑 Prefix 全链。该 softmax 布局仅为后续实验设计，尚未实现或声称收益。转换复用已有约 2.5 ms 局部潜力，但未解决尾延迟不稳定前不默认启用。不改电源/时钟/权限/模型/采样步数，不放宽数值和尾延迟门禁。

### 当前完成边界

A、B1 Expert 单层、Prefix 单层、Expert 16 层常驻链、完整 Expert velocity-step 及固定形状 ViT 已分配图复用已通过数值与模型级回归，并分别完成未插桩测量。各轮 trace 分开保留，不混算计数或时长。CUDA 默认使用 Prefix 单层、完整 Expert step 和固定形状 ViT 图复用；CPU 默认路径仍 staged。正常构建不需要任何 SmolVLA 专用选项/选择变量，各诊断开关仅用于对照。没有合并 CPU 时间 embedding、Euler 或 Prefix→Expert KV bridge。

已新增完成精确 RGB 坐标复用、固定视觉 softmax 寄存器复用及 CC870 连续单消费者 BF16 round-trip 单 kernel，分别按上节独立验收；非连续/多消费者/可观察 BF16 中间值仍保留原 cast。

未完成项：Prefix 层链和阶段间 device bridge、十步常驻/GPU Euler，C 其他 cast/elementwise/probe 融合、native-GQA/GEMM tuning，D vision 输入 staging/拷贝消除、batching/部署。ViT FA 的资格验证失败，保持关闭。上一轮 vision-reuse A/B 约 410.2 ms，ViT 约占 60%、Expert 约 27%、Prefix 约 13%（2026-11-05 独立 profiling 结果见上节，不跨实验计算加速比）；主机构图优化收益已趋小，后续应分别量化 vision 输入 staging/拷贝和 GPU 算子开销，再评估 cast/probe 融合，不重复实现已完成的固定图复用或完整 Expert step。

历史 A/B1 汇总见本会话 `tmp/smolvla-cuda-20261104-1219/optimization/RESULTS.md` 与其 `execution-summary.json`；本轮 Prefix / FA 结果见 `tmp/smolvla-cuda-20261104-1219/post-b1/RESULTS.md`、`final-ab-summary.json`、`validation.json` 及 `vision-fa/qualification.json`；本轮 Expert 层链为 `tmp/smolvla-cuda-20261104-1219/expert-chain/RESULTS.md`、`validation.json`、`final-ab-summary.json`、`scheduler-buffers.json` 和两模式 CUDA trace。两个 GGUF 的 SHA256 与原始记录一致，测试 server 已退出，`git diff --check` 通过；未提交、未推送，保留原先两项 worktree 路径删除。

