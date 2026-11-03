# SmolVLA 适配状态与检查点调查

**状态：CPU 策略已接入现有 foreground HTTP 服务。** 输入状态和输出动作均处于归一化空间，具体机器人需要匹配的统计信息。

专用模型加载器及 CPU 前缀、动作专家和投影实现位于 `src/models/smolvla.cpp`；预处理、视觉和策略编排保留在 `src/jetson_pi_smolvla.cpp`。策略通过专用 llama 模型加载器加载并持有前缀及专家权重，计算图借用这些 CPU 权重，并使用独立的 ggml CPU backend scheduler 执行；处理器统计不作为模型权重加载。运行时会拒绝不支持的策略配置。此实现按阶段建立多个图，并复用 scheduler、已分配计算图及输入输出 buffer；prefix 长度等图形状变化时重新构建。中间结果和 KV 仍经 host 传递，不使用 `llama_context` 自身的 scheduler，也未实现 GPU 路径；普通 Llama graph 不是完整的 SmolVLA 策略图。同一个 policy 实例不可并发调用。独立 reference 工具仍可直接从 GGUF 加载权重。

迁移后使用 `py312` 中的 LeRobot 从 `../models/smolvla_base` 重新生成固定参考数据，对照已有 F32 GGUF：完整策略最大绝对误差为 `0.019934118`、余弦相似度为 `0.999987921`，通过现有验证阈值。该结果仅覆盖固定 CPU 输入，动作仍处于归一化空间。

## 本地检查点

`../models/smolvla_base/config.json` 声明了三个 3×256×256 相机输入、6 维状态与 6 维动作。模型将状态和动作填充至 32 维，动作块为 50 步，通过 10 步 flow matching 采样；VLM 为 SmolVLM2-500M-Video-Instruct，使用 16 层和交叉注意力动作专家。图像缩放配置为 512×512，语言最大长度为 48。

`model.safetensors` 含 500 个张量，包括视觉编码器、VLM、`lm_expert`、状态和动作投影。现有 SmolVLM 转换仅针对 VLM，不能由此得到完整策略；SmolVLA 还需要动作专家的权重转换及匹配的执行图、掩码、KV 缓存和迭代采样逻辑。HTTP 集成应复用已有 foreground 会话和接口，并明确图像顺序、状态和动作格式。

配置要求状态和动作按均值及标准差归一化。检查点的 preprocessor 和 postprocessor 统计文件仅包含 `so100`、`so100-blue`、`so100-red` 三组动作均值和标准差，没有 `observation.state` 统计。它们与配置中的通用 `action` 键也不直接对应。部署时需要明确训练数据及机器人所用的统计信息和映射，不能擅自选用其中一组。输出动作也应清楚标明是否已反归一化。

## 视觉编码器转换

现有转换器可复用 SmolVLM 视觉及 connector 映射。`--smolvla-backbone` 指向与策略配置一致的本地 SmolVLM2 快照，目录应包含 `config.json` 和 `preprocessor_config.json`：

```bash
python convert_hf_to_gguf.py ../models/smolvla_base \
  --mmproj --smolvla-backbone /path/to/SmolVLM2-500M-Video-Instruct \
  --outfile tmp/smolvla-session/mmproj-smolvla.gguf --outtype f32
```

转换读取策略检查点内的视觉权重，骨干目录仅提供配置。F32 导出便于与 BF16 源权重逐项比较。主模型可按下述命令转换，并通过现有 foreground 服务运行。模型预处理使用 SmolVLA 的 resize/pad 和像素归一化规则，不能直接使用普通 SmolVLM 图像切片行为。

## 主模型转换

```bash
python convert_hf_to_gguf.py ../models/smolvla_base \
  --smolvla-backbone /path/to/SmolVLM2-500M-Video-Instruct \
  --outfile tmp/smolvla-session/smolvla-main.gguf --outtype f32
```

主 GGUF 使用 `smolvla` 架构名，包含截断到策略配置层数的 VLM 骨干、动作专家、状态/动作投影、BPE tokenizer、策略配置和处理器统计。`smolvla.tensor_schema_version=1` 标识当前张量命名；没有此字段的旧文件也使用同一布局。转换器会校验当前 CPU 路径支持的策略配置，主模型输出支持 `f32`、`f16` 和 `bf16`，不支持量化输出。两个 GGUF 可用于 CPU foreground 服务。模型使用归一化状态并返回归一化动作；机器人数据集的状态及动作统计需要由调用方正确处理。

## 图像预处理验证

C++ 预处理接受 RGB 字节，进行双线性缩放，在左侧和顶部补零，然后映射到 `[-1,1]`。可使用 LeRobot 对照：

```bash
cmake --build build --target smolvla-preprocess-reference
python tools/smolvla-reference/validate_preprocess.py build/bin/smolvla-preprocess-reference
```

该检查需要 PyTorch、NumPy 和 LeRobot。`smolvla_vision` 将该预处理与现有视觉编码器组合，支持当前 base 检查点的 512×512 CPU 图像路径。可将组合输出与 LeRobot 比较：

```bash
cmake --build build --target smolvla-vision-reference
python tools/smolvla-reference/validate_vision.py build/bin/smolvla-vision-reference \
  ../models/smolvla_base tmp/smolvla-session/mmproj-smolvla.gguf
```

视觉组件已由完整 CPU 策略调用；视觉组件本身不生成动作。

## 动作策略 CPU 参考数据

使用本地 SmolVLM2 配置和 tokenizer，在 CPU 上导出固定三路图像、指令、已归一化状态和固定噪声的参考数据：

```bash
HF_HUB_OFFLINE=1 python tools/smolvla-reference/export_policy_reference.py \
  --checkpoint ../models/smolvla_base --backbone /path/to/SmolVLM2-500M-Video-Instruct \
  --output tmp/smolvla-session/policy-reference
```

`reference.safetensors` 包含图像嵌入、前缀与注意力掩码、每层 KV 缓存、10 步的输入、suffix、专家输出和速度，以及最终的 50×6 动作。数据为合成输入，状态与动作均处于归一化空间；未应用缺失或有歧义的机器人统计。生成器还将手动记录的积分过程与 LeRobot `sample_actions` 的结果对照。可用 `compare_reference.py` 比较后续 C++ 运行时生成的同名 safetensors 张量。

## 动作和时间输入投影

`smolvla_action_input` 使用 GGML 实现固定 50×32 动作输入、720 维时间嵌入和动作时间 MLP。它仅实现动作专家的输入部分，尚不生成动作。当前组件需要 CPU 后端、`chunk_size=50`、`max_action_dim=32` 和周期范围 `[0.004, 4.0]`，加载时会验证配置及权重形状。可对照前述参考数据：

```bash
cmake --build build --target smolvla-action-input-reference
python tools/smolvla-reference/validate_action_input.py \
  build/bin/smolvla-action-input-reference tmp/smolvla-session/smolvla-main.gguf \
  tmp/smolvla-session/policy-reference/reference.safetensors --max-abs 0.0001
```

动作专家最后的 `720 → 32` 速度投影也由同一组件提供，可逐步验证：

```bash
cmake --build build --target smolvla-action-output-reference
python tools/smolvla-reference/validate_action_output.py \
  build/bin/smolvla-action-output-reference tmp/smolvla-session/smolvla-main.gguf \
  tmp/smolvla-session/policy-reference/reference.safetensors --max-abs 0.0001
```

## 状态投影与前缀掩码

状态投影接受已归一化且填充至 32 维的状态，输出 960 维 token；前缀掩码为图像、语言和状态 token 构建可见性及位置 ID。运行参考检查：

```bash
cmake --build build --target smolvla-state-reference smolvla-prefix-mask-reference
python tools/smolvla-reference/validate_state.py build/bin/smolvla-state-reference \
  tmp/smolvla-session/smolvla-main.gguf tmp/smolvla-session/policy-reference/reference.safetensors --max-abs 0.0001
python tools/smolvla-reference/validate_prefix_mask.py build/bin/smolvla-prefix-mask-reference \
  tmp/smolvla-session/policy-reference/reference.safetensors
```

前缀掩码要求普通图像/语言 token 不能看到状态 token，而状态 token 可以看到此前有效 token。已有通用 Llama 非因果模式不能单独表达该规则，执行图仍需要支持这一掩码。

## Flow matching 积分

`smolvla::euler_integrate` 接收初始噪声和速度回调，按 `t = 1 - step / num_steps`、`dt = -1 / num_steps` 更新动作。可使用参考 fixture 中的速度验证调度与数值更新：

```bash
cmake --build build --target smolvla-flow-reference
python tools/smolvla-reference/validate_flow.py build/bin/smolvla-flow-reference \
  tmp/smolvla-session/policy-reference/reference.safetensors
```

此检查回放 LeRobot 的速度，不执行专家 Transformer，也不代表完整推理已实现。

## 后缀掩码与 RoPE

后缀自注意力掩码允许动作 token 访问有效前缀和当前及此前的动作 token；交叉注意力只访问有效前缀。`smolvla_input::build_suffix_attention` 构造这两种掩码及绝对位置，可使用以下命令对照 LeRobot：

```bash
cmake --build build --target smolvla-suffix-mask-reference
python tools/smolvla-reference/validate_suffix_mask.py build/bin/smolvla-suffix-mask-reference \
  tmp/smolvla-session/policy-reference/reference.safetensors
```

SmolVLA 自身的 RoPE 使用基数 10000，不能沿用 SmolVLM2 配置的 100000。专家 Q/K 权重保留分半排列，VLM Q/K 已按照 Llama 转换规则重排。专家侧 GGML RoPE 可通过 `python tools/smolvla-reference/validate_rope.py build/bin/smolvla-rope-reference` 验证。更新转换脚本后必须重新转换主 GGUF；旧主文件包含错误的 RoPE 元数据。

## 注意力基础运算

可对照 LeRobot 分组查询注意力公式：

```bash
cmake --build build --target smolvla-attention-reference
python tools/smolvla-reference/validate_attention.py build/bin/smolvla-attention-reference
```

该检查覆盖 15 个查询头、5 个 KV 头和全遮罩查询，但不加载策略层权重或运行完整 Transformer。

## 专家末端归一化

复用 GGML RMSNorm，并保留 LeRobot BF16 的中间舍入。验证固定参考轨迹：

```bash
cmake --build build --target smolvla-expert-norm-reference
python tools/smolvla-reference/validate_expert_norm.py build/bin/smolvla-expert-norm-reference \
  tmp/smolvla-session/smolvla-main.gguf tmp/smolvla-session/policy-reference/reference.safetensors
```

## 专家前馈网络

使用 GGML 实现单层专家 MLP，保持 BF16 中间舍入，可对照 16 层和 10 步参考轨迹：

```bash
cmake --build build --target smolvla-expert-mlp-reference
python tools/smolvla-reference/validate_expert_mlp.py build/bin/smolvla-expert-mlp-reference \
  tmp/smolvla-session/smolvla-main.gguf tmp/smolvla-session/policy-reference/reference.safetensors
```

该组件仅计算 MLP，不包含注意力、残差或完整策略。

## 专家层后半部分

已将注意力输出投影、残差、RMSNorm 和 MLP 串接。以下验证使用 LeRobot 提供的注意力输出与隐藏状态，不执行注意力本身：

```bash
cmake --build build --target smolvla-expert-finish-reference
python tools/smolvla-reference/validate_expert_finish.py build/bin/smolvla-expert-finish-reference \
  tmp/smolvla-session/smolvla-main.gguf tmp/smolvla-session/policy-reference/reference.safetensors
```

## 专家 Q/K/V 投影

专家的偶数层使用自注意力，奇数层使用交叉注意力。交叉注意力的 K/V 投影接受 320 维前缀缓存；此检查点奇数层 K/V 权重为 F32，偶数层及 Q 投影为 BF16。可验证所有层和去噪时刻：

```bash
cmake --build build --target smolvla-expert-qkv-reference
python tools/smolvla-reference/validate_expert_qkv.py build/bin/smolvla-expert-qkv-reference \
  tmp/smolvla-session/smolvla-main.gguf tmp/smolvla-session/policy-reference/reference.safetensors
```

该投影验证使用 LeRobot 提供的隐藏状态和前缀缓存；完整专家层另有组合验证。

## 完整动作专家

CPU 专家已串联 16 层 Q/K/V、RoPE、注意力、残差、MLP 及最终 RMSNorm。该组件接收前缀 KV 缓存和动作 suffix；VLM 前缀执行可生成缓存并已接入 CPU 服务。验证可使用 LeRobot 缓存：

```bash
cmake --build build --target smolvla-expert-reference smolvla-expert-layer-reference
python tools/smolvla-reference/validate_expert.py build/bin/smolvla-expert-reference \
  tmp/smolvla-session/smolvla-main.gguf tmp/smolvla-session/policy-reference/reference.safetensors
python tools/smolvla-reference/validate_expert_layer.py build/bin/smolvla-expert-layer-reference \
  tmp/smolvla-session/smolvla-main.gguf tmp/smolvla-session/policy-reference/reference.safetensors
```

## 基于参考前缀缓存的完整动作采样

`smolvla_expert::sample` 已串联动作输入投影、16 层专家、速度投影及 10 步 Euler 积分。它目前需要外部提供前缀 KV 缓存，输出 50×32 归一化动作；本检查点的有效动作维度为前 6 维。

```bash
cmake --build build --target smolvla-sample-reference
python tools/smolvla-reference/validate_sample.py build/bin/smolvla-sample-reference \
  tmp/smolvla-session/smolvla-main.gguf tmp/smolvla-session/policy-reference/reference.safetensors
```

参考输入为固定噪声和 LeRobot 前缀缓存；这不覆盖前缀计算、HTTP 服务和机器人动作反归一化。

## VLM 前缀与动作采样

CPU 前缀执行接受已构造的 `prefix_embedding`、注意力掩码和位置 ID，生成 16 层 KV 缓存；该缓存可直接交给动作采样器。以下检查使用 LeRobot 生成的前缀嵌入及固定噪声：

```bash
cmake --build build --target smolvla-prefix-reference smolvla-sample-reference
python tools/smolvla-reference/validate_prefix.py build/bin/smolvla-prefix-reference \
  build/bin/smolvla-sample-reference tmp/smolvla-session/smolvla-main.gguf \
  tmp/smolvla-session/policy-reference/reference.safetensors
```

此检查直接使用参考前缀嵌入；独立的端到端和 HTTP 检查覆盖 RGB 图像和文本。

## CPU 策略端到端验证

`smolvla_policy` 接收三路 RGB、指令、6 维已归一化状态和固定 50×32 噪声，输出 50×6 归一化动作；调用者需根据机器人数据集处理状态归一化和动作反归一化。该组件也已接入 foreground HTTP 服务。

```bash
cmake --build build --target smolvla-policy-reference
python tools/smolvla-reference/validate_policy.py build/bin/smolvla-policy-reference \
  tmp/smolvla-session/smolvla-main.gguf tmp/smolvla-session/mmproj-smolvla.gguf \
  tmp/smolvla-session/policy-reference/reference.safetensors tmp/smolvla-session/policy-reference/metadata.json
```

tokenizer 可使用 `tools/smolvla-reference/validate_tokenizer.py` 对照本地 SmolVLM2 tokenizer，包含左侧截断及特殊 token。

## Foreground HTTP 服务

CPU 构建需要 `GGML_CPU=ON` 且 `GGML_BACKEND_DL=OFF`。设置 `PI_MODEL=pi0` 或 `pi05` 会与 SmolVLA 冲突；使用默认值或 `PI_MODEL=auto`。使用已有 foreground 路由，不新增 Web 接口：

```bash
./build/bin/llama-server -m tmp/smolvla-session/smolvla-main.gguf \
  --mmproj tmp/smolvla-session/mmproj-smolvla.gguf -ngl 0 -np 1 \
  --host 127.0.0.1 --port 8080
curl -X POST http://127.0.0.1:8080/foreground/reset
# 分别提交三个摄像头图像，顺序对应 camera1、camera2、camera3
curl -X POST http://127.0.0.1:8080/foreground/image -H 'Content-Type: application/json' -d '{"path":"/path/to/camera1.png"}'
curl -X POST http://127.0.0.1:8080/foreground/image -H 'Content-Type: application/json' -d '{"path":"/path/to/camera2.png"}'
curl -X POST http://127.0.0.1:8080/foreground/image -H 'Content-Type: application/json' -d '{"path":"/path/to/camera3.png"}'
curl -X PUT http://127.0.0.1:8080/foreground/state -H 'Content-Type: application/json' -d '{"state":[0,0,0,0,0,0]}'
curl -X POST http://127.0.0.1:8080/foreground/infer -H 'Content-Type: application/json' -d '{"text":"pick up the object"}'
```

`state` 为 6 个已归一化数值，可提交 JSON 数组或逗号分隔字符串（兼容现有 Python 客户端）；返回的 `action_final` 是 50×6 归一化动作，`action_normalized` 为 true。成功推理后待处理图像和文本被清除，状态保留至 reset。默认每次使用随机正态噪声；验证时可设置 `SMOLVLA_ACTION_NOISE_BIN` 指向恰好 50×32 个 float32 的固定噪声文件。切勿在没有匹配统计信息和机器人适配的情况下直接将归一化动作发送到机器人。

固定噪声 HTTP 验证：

```bash
SMOLVLA_ACTION_NOISE_BIN=$PWD/tmp/smolvla-session/noise ./build/bin/llama-server \
  -m tmp/smolvla-session/smolvla-main.gguf --mmproj tmp/smolvla-session/mmproj-smolvla.gguf \
  -ngl 0 -np 1 --host 127.0.0.1 --port 8080
python tools/smolvla-reference/validate_http.py http://127.0.0.1:8080 \
  tmp/smolvla-session/policy-reference/reference.safetensors \
  tmp/smolvla-session/policy-reference/metadata.json tmp/smolvla-session/http-fixture
```

## 验证记录

- 使用 `/home/lyf/miniconda3/envs/py312/bin/python` 运行 `convert_hf_to_gguf.py` 对该检查点转换，失败于 `Failed to detect model architecture`；这是未指定骨干目录时的历史检查结果；现在可按上面的命令导出并通过 CPU 服务运行。
- 现有 foreground HTTP 路由的 CPU 固定噪声动作对照、状态/图像校验和 reset 均通过；最大绝对误差 0.019935。
- 三路 RGB、文本、已归一化状态与固定噪声的 CPU 端到端动作对照：最大绝对误差 0.019935，余弦相似度 0.9999879。
- 使用 LeRobot 前缀嵌入、C++ 前缀缓存及固定噪声，10 步动作采样的 50×6 输出最大绝对误差 0.01849、余弦相似度 0.9999776。
- 使用 LeRobot 前缀缓存的完整 10 步采样，50×6 归一化动作最大绝对误差 0.009813，余弦相似度 0.9999961。
- 16 层专家完整串联，使用 LeRobot 前缀缓存对 10 步参考输入逐步验证，最大绝对误差 0.0625，最低余弦相似度 0.9999736；逐层验证最大绝对误差 0.03125。
- 专家 Q/K/V 投影在 16 层、10 个去噪时刻对照 LeRobot，最大绝对误差 0.015625。
- 专家层后半部分在 16 层、10 步参考轨迹上最大绝对误差 0.03125、平均绝对误差 9.1e-7；输入注意力输出来自 LeRobot。
- 16 层专家 MLP 在 10 步参考轨迹上全部通过对照，最大绝对误差 0.0078125。
- 专家末端 BF16 RMSNorm 参考对照：10 步中 9 步完全相同，另一步最大误差 0.00390625、平均误差 2.58e-7。
- GGML 分组查询注意力的四组 CPU 对照（包括全遮罩查询）最大绝对误差 8.95e-7；该注意力运算已串入专家层。
- 后缀自注意力、交叉注意力掩码及位置与 LeRobot 精确一致；专家 RoPE 随机输入最大绝对误差 4.76e-5。
- 用参考速度回放 10 步 Euler 积分，输出与 LeRobot 完全一致；此检查单独验证积分器；完整策略也已接入专家 Transformer。
- 状态投影对照：F32 最大绝对误差 5.96e-8，BF16 最大绝对误差 8.17e-4；241 个前缀位置的注意力掩码与位置 ID 精确匹配。
- 动作输出投影的 10 步对照：F32 最大绝对误差 1.67e-6，BF16 最大绝对误差 0.00421。
- 10 个去噪时刻的动作输入投影均与 LeRobot 对照：F32 权重最大绝对误差 1.67e-6；BF16 权重最大绝对误差 0.00675。
- 固定输入的 LeRobot CPU 参考推理生成包含各层 MLP 中间结果的有限张量，最终动作维度为 1×50×6；手动记录的 10 步积分与 `sample_actions` 一致。已用于 C++ 动作执行对照。
- 主 GGUF 的 314 个张量（302 个非视觉模型张量和 12 个处理器统计张量）与源权重逐项一致；Llama 骨干 Q/K 按现有转换器规则重排。
- 图像预处理对照覆盖七组宽高，最大绝对误差 3.58e-6。将 640×360 输入经预处理和视觉执行图后与 LeRobot 对照，最大绝对误差 0.02062，余弦相似度 0.999999915。
- SmolVLA mmproj 的 198 个视觉与 connector 张量全部与源权重数值相等（BF16 转 F32）；现有 `clip_init` 在 CPU 上成功加载，投影维度为 960。使用相同的已归一化 512×512 图像，视觉输出均为 64×960，最大绝对误差 0.01824，平均绝对误差 0.001386；此对比不涉及图像预处理。
- 原始权重 GGUF 共 512 个张量；与 safetensors 逐个比较名称、形状、精度和字节均一致。
- 同一环境中用 LeRobot `SmolVLAPolicy.from_pretrained`、CPU、`strict=True` 成功加载检查点权重；这只是加载验证，不是推理验证。初始化还尝试获取上游 SmolVLM tokenizer 资源，网络读取出现超时重试。
- 现有 pi0、pi05 的 CPU C API 回归各运行两次，输出均为有限的 10×32 且两次完全一致；GR00T CPU C API 输出有限的 40×132 动作。此回归验证形状、有限性及 PI 重复性，未与旧版本动作数值逐元素比较。

后续需要基于固定输入与固定噪声，将转换后的 C++ CPU 输出同 LeRobot 参考实现逐阶段比较，并对三个已有模型运行实际 foreground 推理回归。

参考：[SmolVLA 介绍](https://infinity4b.github.io/learning-record/vla/SmolVLA)、[LeRobot](https://github.com/huggingface/lerobot)。
