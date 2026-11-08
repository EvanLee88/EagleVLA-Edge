# SmolVLA CUDA 分阶段 profiling

## 2026-11-06：视觉 Q/K/V 输入转换复用（未采纳，已回退）

本轮候选通过严格数值/模型/API/HTTP 回归，但**最终原仓库 ABBA 的总 p95 回退，未通过性能门禁，已回退本轮实现和新增测试**。生产保留此前验收的 RGB、softmax 与 BF16 round-trip 优化，不把该候选记作已交付加速。

候选仅共享固定 strict CUDA Idefics3 / Orin 的 separate Q/K/V F32→F16 输入转换，保留 GEMM 累加/输出精度；另包含运行时 compute override 的 capture 失效修正。实际构建 NVTX OFF ABBA，每进程 201 次、每模式 **400 warm**：

| 阶段 | 旧 mean ms | 被拒绝候选 mean ms |
|---|---:|---:|
| 三路 ViT | 224.428 | 221.868 |
| Prefix prepare | 1.946 | 1.999 |
| LLM Prefix | 39.205 | 39.616 |
| 十步 Expert | 99.343 | 99.453 |
| 完整 policy | 365.327 | 363.340 |

总 mean −1.987 ms、median **364.866→362.400 ms**，但总 **p95 366.987→369.455 ms（+2.468 ms）**；排除前两次 p95 仍 **366.981→369.415 ms**。ViT mean 改善约 2.56 ms、p95 224.828→222.423 ms，但不能以局部改善替代全链路尾延迟门禁。此前扩展隔离轮 total p95 368.861→364.464 ms，不能覆盖最终实际构建的回退；各轮完整保留，不反复挑选通过的轮次。

数值通过不等于性能通过：35,389,440 个 GEMM 输出值、三路 embedding/六组 policy、CPU policy、API/vision、CUDA 10/10 与 CPU 6/6 CTest、HTTP 15 项（max_abs=0）和同实例精度切换均通过。trace 的 GEMM kernel 名称/计数不变，每 policy 净少 72 个 conversion kernel，仅作 dispatch 证据。原源码哈希和保存的 CUDA binary/libs 已恢复，CPU 已重建，恢复后完整回归通过（CUDA 8/8、CPU 6/6 CTest，三路 embedding/六组 CUDA policy、CPU policy、两 backend API/vision、HTTP 15 项 / max_abs=0）；发现的旧 capture 配置切换问题保留在隔离产物，未声称已修复生产版本。详见[深度报告第 10 节](smolvla-orin-latency-analysis-20261105.md#10-2026-11-06视觉-qkv-输入转换复用)。产物：`tmp/smolvla-cuda-20261104-1219/vision-convert-20261106/`。

## 2026-11-06：精确 BF16 round-trip 单 kernel 融合

已在 CC870 默认启用连续、相邻、单外部消费者的真实 F32→BF16→F32 cast 融合，保留 BF16 中间舍入、finite probe 和模型图；OUTPUT/view/多消费者/部分内存重叠走原路径，允许精确同址。无新 API/选项，CPU 和其他 CUDA 架构不变。详见 [深度报告第 9 节](smolvla-orin-latency-analysis-20261105.md#9-2026-11-06保留-bf16-舍入的双-cast-单-kernel-融合)。

原仓库实际构建与保存的本轮旧 binary/libs **A→B→B→A**，每进程 101 次、剔首轮，每模式 **200 warm**，NVTX OFF，未锁频、不与构建/其他 GPU 实验重叠，全部动作逐位一致：

| 阶段 | 旧 mean ms | 新 mean ms |
|---|---:|---:|
| 三路 ViT | 224.299 | 224.672 |
| Prefix prepare | 1.981 | 2.000 |
| LLM Prefix | 44.790 | **40.730** |
| 十步 Expert | 110.089 | **100.354** |
| 完整 policy | **381.560** | **368.157** |

完整 policy mean **−13.403 ms / −3.51%**，median **380.354→367.217 ms**，p95 **390.928→374.398 ms**；Prefix **−9.07%**、Expert **−8.84%**。两组 control total=382.392/380.727 ms，两组新版本=368.230/368.084 ms，收益大于 control 漂移；排除前两次 total mean **381.178→367.727 ms**。ViT/prepare 未优化，其波动不作收益归因。距 250 ms 仍差 **118.157 ms**，尚未达标；不与下节另一轮旧绝对值跨实验算加速比。

直接 CUDA 旧双 kernel 对照 **42,665,412** 输入 bits 零差异（含全部 NaN/subnormal 编码），永久图级回归 **28,908,800** 个 F32 输出值逐位一致，另检查 BF16 OUTPUT 和 alias 安全。六组 CUDA policy、三路 embedding、CPU 完整 policy、两 backend API/vision reuse、CUDA 8/8 与 CPU 6/6 CTest、HTTP 15 项（max_abs=0）均通过。三次 policy trace 命中 **9222 次**新 kernel，动作逐位一致，仅作 dispatch 证据。模型哈希不变，server 已退出、diff check 通过；CUDA memcheck 未运行。产物：`tmp/smolvla-cuda-20261104-1219/bf16-round-20261106/`。下一优先项为视觉 conversion/operand 复用及 Prefix 全链，FA 仍关闭。

## 2026-11-06：固定形状 vision softmax 寄存器复用

已默认应用严格限定 Orin CC 8.7、F32、1024×1024×12×1、无 mask/sinks 的寄存器中间值版本，保留 1024 线程、算术和 MAX/SUM 规约树；其他调用走旧 kernel，无新选项，FA 仍关闭。详见 [深度报告第 8 节](smolvla-orin-latency-analysis-20261105.md#8-2026-11-06固定形状-vision-softmax-寄存器复用)。

原仓库实际构建与保存的本轮旧二进制/库交错 **A→B→B→A**，每进程 101 次、剔首轮，每模式 **200 warm**，NVTX OFF、未锁频，所有动作逐位一致：

| 阶段 | 旧 mean ms | 新 mean ms |
|---|---:|---:|
| 三路 ViT | 226.143 | 224.203 |
| Prefix prepare | 1.964 | 1.965 |
| LLM Prefix | 45.076 | 44.173 |
| 十步 Expert | 110.012 | 109.413 |
| 完整 policy | **383.599** | **380.156** |

完整 policy mean **−3.443 ms / −0.90%**，median **382.814→379.631 ms**，p95 **387.648→382.659 ms**；排除前两次 mean **383.234→379.778 ms**。ViT mean 改善 **1.941 ms / 0.86%**，两个 control 与两个新版本均同向。Prefix/Expert 未优化，其约 1.50 ms 变化不归于 softmax；独立候选轮 ViT 也仅改善约 2.04 ms，故不把全部总差额解释为 kernel 收益。距 250 ms 尚差 **130.156 ms**，未达目标；不与下节上一轮绝对值跨实验算收益。

直接 kernel 对照 150,994,944 值逐位一致（含 NaN payload），独立 kernel 约 −2.35%。新增永久 CUDA softmax test；实际六组 CUDA policy、三路 embedding、CPU 完整 policy、两 backend 的 API/vision reuse、CUDA 7 项和 CPU 6 项 CTest、HTTP 15 项（max_abs=0）通过。三次完整 policy trace 命中 **108 次**新 kernel，仅作 dispatch 证据；未将 profiler 时间混入 ABBA。GGUF 哈希不变，server 已退出，diff check 通过。产物：`tmp/smolvla-cuda-20261104-1219/softmax-20261106/`。

## 2026-11-06：精确 RGB 预处理坐标复用

已按 P0/E6a 实现单线程横向坐标/源偏移复用和行地址预计算，保持原插值/归一化运算顺序。默认启用，无新编译/运行时开关，CPU/CUDA 均适用。完整实现、验证和原始数据见 [深度报告第 7 节](smolvla-orin-latency-analysis-20261105.md#7-2026-11-06精确-rgb-坐标复用已实现并验收)。

原仓库实际构建与保存的旧二进制/库 NVTX OFF 交错 A→B→B→A，各进程 31 次排除首轮、每模式 60 warm：

| 阶段 | 旧 mean ms | 新 mean ms |
|---|---:|---:|
| 三路 ViT | 244.540 | 226.580 |
| Prefix prepare | 1.961 | 1.965 |
| LLM Prefix | 45.268 | 45.524 |
| 十步 Expert | 109.856 | 109.982 |
| 完整 policy | **402.024** | **384.456** |

完整 policy mean 下降 **17.569 ms / 4.37%**，median **400.852→382.497 ms**、p95 **405.864→395.273 ms**；三路 ViT mean 下降 **7.34%**。没有改变 GPU 算子、采样步数或精度，不将 Prefix/Expert 波动归为收益。距 250 ms 仍差约 134.456 ms，尚未达标。

永久 preprocess test 实测 1402 组 / 72,411,684 float 逐位一致；六组 CUDA policy、三路 embedding、CUDA/CPU policy API 和 vision reuse、两构建各 6 项 backend/preprocess CTest、CPU-only 完整 policy、HTTP 15 项（max_abs=0）通过。CPU input-only ASan+UBSan 检查通过，不代表 CUDA memcheck。模型哈希不变，未锁频或改设备权限。产物在 `tmp/smolvla-cuda-20261104-1219/preprocess-20261106/`；旧实验表保留，不跨轮计算加速比。

## 2026-11-05 Orin 深度 profiling 更新

完整报告、LeRobot 比较契约、逐算子拆分和下一阶段实验见 [Orin 延迟分析与 250 ms 目标](smolvla-orin-latency-analysis-20261105.md)。本轮未修改生产推理实现，旧表作为历史记录保留。

- 构建结束后独立运行，NVTX OFF，2 个进程各 31 次、各排除首轮，共 60 个 warm：ViT **244.015 ms**、prepare **1.978 ms**、Prefix **45.826 ms**、十步 Expert **109.695 ms**、total **401.915 ms**；total median **401.017 ms**、p95 **419.345 ms**。默认输出与原 CUDA fixture 逐位一致。
- NVTX-only 另测 30 个 warm，total mean=401.848 ms：三路 CPU RGB 预处理 **53.967 ms**，同步视觉 graph compute **179.791 ms**；staging/深拷贝约 **6.96 ms**。CPU DVFS 不可忽略：独立 telemetry 期间 CPU 多数只读采样为 729 MHz；未锁频、未更改电源或权限。
- CUDA Graph node trace 的第二次仍有 26 次 instantiate，因此同时保留排除首轮的 6 次与排除前两次的 5 次。steady trace 每 policy **18,870 kernels**，视觉 softmax **80.066 ms / 36 kernels**、视觉 GEMM convert **28.248 ms**，Prefix/Expert BF16 copy/cast **9.821 / 20.851 ms**。trace wall=502.714 ms，明显有插桩干扰，不能替代上述 401.915 ms 主基线。
- 四种 GEMM 数值/算法诊断均改变全部 300 个公开动作值；强制 BF16/F16 还更慢，全部不采纳。此前 FA 严格资格失败仍有效。Nsight Compute 因性能计数器权限失败，无 occupancy/带宽数据。
- 用户提供的 LeRobot 约 250 ms 尚未完成本机同口径复现；当前 mean 到目标仍差 **151.915 ms**，需要约 **37.8%** 的端到端下降。缓存命中、单步 Expert 或排除预处理的 GPU 模型时长不可用于替代完整 policy 比较。

原始数据和脚本：`tmp/smolvla-cuda-20261104-1219/comprehensive-20261105/`。与编译重叠的探索测量单独归档，不混入独立基线，也不把不同 DVFS 下的差额解释为代码加速。

## 选择 backend

库 API、reference CLI 和 foreground server 的默认 backend 跟随构建：`GGML_CUDA=ON` 默认 CUDA0，CPU-only 构建默认 CPU。CUDA 默认使用单 Prefix 层常驻图和完整 Expert velocity-step 图（encode→16 层→norm→velocity），`SMOLVLA_EXPERT_STEP=0` 恢复层链对照，`SMOLVLA_EXPERT_CHAIN=0` 优先恢复单 Expert 层常驻对照；Prefix 层间仍有主机传递，时间 embedding 和 Euler 仍在 CPU。只设置 `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON` 即可编译和默认使用 CUDA，无需 SmolVLA 专用选项。`SMOLVLA_BACKEND=cpu|cuda`、`SMOLVLA_DEVICE` 仅作运行时覆盖，`SMOLVLA_PREFIX_RESIDENT=0`、`SMOLVLA_EXPERT_RESIDENT=0` 分别选择 staged 对照。严格 CUDA Idefics3 的固定 512×512 视觉已分配图默认复用，`SMOLVLA_VISION_REUSE=0` 逐图重建用于对照；不缓存像素或 embedding，FA 保持关闭，普通 CLIP/CPU 路径不启用该缓存。CUDA 不可用或算子不支持时失败，不静默回退。下文旧 profiling 表保留当时 staged 路径的历史结果，最新 resident 测量见优化执行记录。

### 默认构建验证（2026-11-04）

在全新目录仅传入两个选项配置成功，并从该目录实际构建、运行完整策略：

```bash
cmake -S . -B build-cuda -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON
cmake --build build-cuda --target smolvla-policy-reference -j4
```

- 未设置任何 `SMOLVLA_*` 环境变量的六组完整 policy 输入，输出与保存的 CUDA 基线逐位一致，policy compute 节点 placement 均为 CUDA。
- 默认 sample 与显式 `SMOLVLA_EXPERT_RESIDENT=1` 的 graph placement 一致，与 staged=0 不同；两种路径输出均与 CUDA 基线逐位一致。`SMOLVLA_BACKEND=cpu` 的输出与 CPU 基线逐位一致。
- 默认值、显式 CPU/CUDA 覆盖、非法 backend、空 device 的 5 项测试，在 CUDA 和独立 CPU-only 新构建中各自全部通过。模型级 sample / public API 默认参数 / graph reuse 回归通过。
- 未设置 backend/device/resident 变量的真实 HTTP 请求通过，动作与 CUDA 基线一致；仅设置固定 noise 文件用于可重复比较，不用于选择 backend。
- 对全新二进制进行 CUDA trace，单次完整策略记录到 **216 次 cudaGraphLaunch**，输出仍与 CUDA 基线逐位一致；无需启用 NVTX 或其他 SmolVLA 专用选项。
- 另在全新 `build-no-options` 目录执行不带任何 `-D` 参数的 `cmake -S . -B ...`，并实际构建和运行完整 policy。默认 `GGML_CUDA=OFF`，无任何 `SMOLVLA_*` 环境变量，83 项 policy graph placement 全为 CPU，动作与保存的 CPU 基线逐位一致；该构建的 5 项 backend 测试也全部通过。证明普通 CPU 构建不需要显式 `-DGGML_CUDA=OFF` 或运行时覆盖。
- 日志及原始结果：`tmp/smolvla-cuda-20261104-1219/default-cuda/`，无参数构建证明为 `no-options-verification.json`。CUDA 不可用测试保持报错，不静默回退 CPU。复用构建目录时要注意 CMake 缓存，删除命令中的 CUDA 参数不会清除先前缓存的 ON。

以下为更显式的 Orin（CUDA arch 87）profiling 构建示例；这些额外选项不是默认 CUDA SmolVLA 的启用条件，其他设备应使用其实际架构：

```bash
B=tmp/smolvla-profile/build-cuda
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_CPU=ON -DGGML_BACKEND_DL=OFF \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$B" --target smolvla-policy-reference llama-server -j4
```

## 计时范围

`smolvla_policy::infer()` 可选接收 `smolvla_policy_timings *`，省略时不读取计时器。结构在入口清零，成功时才发布结果，不使用全局累计状态。调用方需要重新编译以使用新的 C++ 签名。

| 字段 | 范围 |
|---|---|
| `vit_ms` | 三路图像编码总和，包括 RGB resize/pad/normalize、ViT、projector 和阶段传输，不是单路或纯 kernel 时间 |
| `prefix_prepare_ms` | 状态投影、语言 embedding、prefix 组装、mask 和 position 准备 |
| `llm_prefill_ms` | 完整 16 层 prefix KV 计算，当前 base 输入通常共 241 tokens |
| `action_expert_ms` | 全部十步 flow-matching/Euler 采样，包括动作/时间投影、expert、输出投影和主机积分，不是单步时间 |
| `total_ms` | 完整成功 `infer()`，还包含 tokenization、检查与动作提取；不含模型加载、HTTP 请求排队或响应序列化 |

**LLM 阶段是 prefix prefill，不是自回归 decode。** 模型不生成逐 token 文本；先处理图像、语言和状态前缀，再复用 KV 进行动作采样。

使用 `std::chrono::steady_clock` 测量阶段 wall time。当前阶段内部调用同步的 `ggml_backend_sched_graph_compute()`，返回前同步 scheduler 的 backend，并完成主机读回，因此不是仅计量 CUDA launch。没有新增 `cudaDeviceSynchronize()`，避免等待无关设备工作。

阶段时长互不重叠，其和不超过 `total_ms`（打印四舍五入可能有微小偏差）。这是分阶段执行的端到端延迟拆分，不是 GPU kernel 时间拆分；预处理、BF16 舍入、阶段间 host 传输仍存在。

## reference CLI

位置参数不变，三路 RGB 均为 256×256×3 原始字节，state 为 6 个已归一化 float32，noise 为 50×32 个 float32：

```bash
SMOLVLA_BACKEND=cuda SMOLVLA_BENCH_REPEATS=7 \
"$B/bin/smolvla-policy-reference" \
  ../models/smolvla_gguf/f16/smolvla-main.gguf \
  ../models/smolvla_gguf/f16/mmproj-smolvla.gguf \
  /path/to/camera0.rgb /path/to/camera1.rgb /path/to/camera2.rgb \
  /path/to/state.f32 /path/to/noise.f32 \
  'Pick up the red cube and place it in the bowl.' /path/to/actions.f32
```

每轮 stderr 打印以下字段（此处仅展示格式，不是测量值）：

```text
policy infer repeat=2/7 infer_ms=... vit_ms=... prefix_prepare_ms=... llm_prefill_ms=... action_expert_ms=... total_ms=...
policy infer_warm_mean repeats=6 infer_ms=... vit_ms=... prefix_prepare_ms=... llm_prefill_ms=... action_expert_ms=... total_ms=...
```

首轮包含图分配、CUDA Graph warmup 等开销；warm mean 排除第一轮。所有轮使用同一个模型实例。`infer_ms` 为 CLI 外层计时（含输出检查），`total_ms` 为内部 policy 范围。模型加载另外打印 `load_ms`，不参与阶段统计。

## foreground server

```bash
PI_MODEL=auto SMOLVLA_BACKEND=cuda \
"$B/bin/llama-server" \
  -m ../models/smolvla_gguf/f16/smolvla-main.gguf \
  --mmproj ../models/smolvla_gguf/f16/mmproj-smolvla.gguf \
  -ngl 0 --no-mmproj-offload --no-warmup \
  --host 127.0.0.1 --port 8080 -np 1 -c 1024 -t 4 -tb 4
```

专用 SmolVLA policy 的 backend 独立于通用 server 的 `-ngl`；此示例将重复加载的通用模型/projector 留在 CPU，专用 policy 使用 CUDA。通过既有 `/foreground/image`、`/foreground/state`、`/foreground/infer` 接口推理。固定 `SMOLVLA_ACTION_NOISE_BIN` 可复现动作输出。

每次成功请求打印阶段耗时，响应 `timing_breakdown_ms` 中返回上述五项和原有 `handler_total_ms`；handler 时间还包含请求解析和锁等待。其他动作字段及 foreground 会话行为不变。

## 数值与部署边界

相同 F16 GGUF 的六组固定完整输入 CUDA/原 CPU 迁移验证通过预设 `max_abs ≤ 0.05`、`cosine ≥ 0.999`；CPU 回归逐位一致。该阈值不是机器人控制安全容差。

独立组件的短序列合成输入跨 backend 测试仍有 29/53 项超差，主要是深层 prefix K/V 和 aggregate expert。同 backend 重复、shape 重建、fresh handle 与跨线程检查通过。短序列 F16 的 CPU NEON half-accumulation 与 CUDA MMF float-accumulation 不同，不能据完整样例通过推断所有层、所有输入等价。未运行本次 CUDA 的独立 LeRobot/PyTorch oracle。

本机 Compute Sanitizer 因 `GPU debugging features are disabled` 无法完成设备 memcheck；不能声称其通过。没有实机机器人动作验证。不要直接将归一化动作发送给机器人。

## 本机测量

2026-11-04，直接从原仓库构建并运行：Orin、CUDA 13.2、架构 87、Release、4 CPU threads，模型为 `../models/smolvla_gguf/f16`。固定三路坐标图案、六维状态、噪声及指令，CPU→CUDA 顺序测量；每个 backend 加载一次、同实例推理 7 次，排除首轮，统计余下 6 次。计时不调功耗、不锁频，结果仅代表所测设备、模型、固定输入和运行时状态。

| 阶段 | CPU warm mean ms | CUDA warm mean ms | CUDA warm median ms |
|---|---:|---:|---:|
| ViT（三路合计） | 6701.116 | 261.867 | 261.411 |
| Prefix prepare | 0.766 | 2.135 | 1.989 |
| LLM prefix prefill | 745.254 | 209.913 | 209.169 |
| Action expert（十步合计） | 1387.788 | 1058.970 | 1060.760 |
| 完整 policy | 8835.142 | 1533.278 | 1535.745 |

完整 policy warm mean 加速约 **5.76×**；CUDA 下 action expert 为主要耗时阶段。上述时间不包含模型加载，不是从 Nsight instrumentation 运行采样。单独完成了原仓库 policy 的 Nsight CUDA trace。

原仓库 CPU/CUDA 六组完整输入回归、A→B→A 复用、计时开启/关闭输出逐位一致、失败清零、HTTP 动作与阶段 JSON 检查均通过。所有阶段值有限非负，阶段和不超过内部 total。旧的组件数值超差与 memcheck 不可用限制仍保留。

本会话复现脚本和原始样本保存在 `tmp/smolvla-cuda-20261104-1219/`（被 Git 忽略，不是仓库必需依赖）：

- `scripts/profile_stages.py`：顺序 CPU/CUDA 7 轮测量与 warm mean/median/min/max 汇总。
- `bench/repo-stage-profiling.json`：每个阶段全部 warm 样本和统计。
- `bench/repo-stages-cpu.log`、`bench/repo-stages-cuda.log`：实际打印日志。
- `results/REPO-PROFILING.md`、`results/repo-profiling-validation.json`：验证报告。
- `bench/repo-cuda-policy.nsys-rep`：独立采集的实际 CUDA trace。

在保留本会话 fixtures 的环境中可重跑：

```bash
S="$PWD/tmp/smolvla-cuda-20261104-1219"
python3 -I "$S/scripts/profile_stages.py" "$S" \
  "$S/build-repo-cuda/bin/smolvla-policy-reference"
```

`build-repo-cuda` 的 `CMAKE_HOME_DIRECTORY` 为原仓库根目录，而非此前的 tmp 源码副本。

## 后续优化

上述表格保留为 CUDA 适配后的历史基线，不随优化结果覆盖。优化顺序、数值门禁、可选 NVTX 标记和前后实测见 [SmolVLA CUDA 优化方案与执行记录](smolvla-cuda-optimization-plan.md)。该文档的“B1 后的调优优先级”补充了 resident 路径下 Prefix、ViT、Expert 的 trace 依据。本轮 Prefix 单层常驻已完成：同一二进制交错 A/B，Prefix mean 201.108→49.694 ms，完整 policy 630.782→474.904 ms，动作逐位一致；CUDA 默认启用。ViT FA 已独立验证，因 embedding/动作不满足严格等价而保持关闭。随后 Expert 16 层链也已完成并默认启用：新一轮同二进制交错 A/B，Expert mean 170.846→118.941 ms，完整 policy 469.983→420.930 ms（下降 10.44%），六组动作逐位一致；此处是独立实验，不与 Prefix 轮不同状态的 474.904 ms 混算收益。此后完整 Expert velocity-step 通过 22 项数值/复用验证并默认启用：独立未插桩 ABBA，十步 Expert 120.412→112.098 ms（−6.90%），完整 policy 423.333→415.406 ms（−1.87%）；六组原 CUDA 输出逐位一致，无变量默认 CLI/API、HTTP 15 项与 CPU-only 真实推理回归均通过。三次完整推理独立 trace 的 cudaGraphLaunch 总数 176→89（含首轮，不作为 warm 阶段计数）。241-token Expert scheduler buffer 略增 6,400 bytes，不代表总显存节省。固定形状 ViT 已分配图复用随后也通过 24 项数值/集成验证并默认启用：本轮独立 ABBA，三路 ViT mean 250.314→245.841 ms（−1.79%），完整 policy 413.363→410.217 ms（−0.76%），三路 embedding 和六组动作逐位一致。它只消除重复的主机构图/分配流程，不改 GPU 数学、FA、像素 staging 或三路完整计算。原始样本、门禁与未完成项见该文档执行记录及 `tmp/smolvla-cuda-20261104-1219/{post-b1,expert-chain,expert-step,vision-reuse}/`，不覆盖本页历史测量，也不跨实验混算收益。
