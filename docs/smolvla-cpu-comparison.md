# SmolVLA CPU 动作对照测试

本测试通过本地 `llama-server` 的 foreground HTTP 接口比较 EagleVLA-Edge CPU 策略与本机 LeRobot `SmolVLAPolicy`，使用 `smolvla_base` 检查点和 F32 转换的主 GGUF、视觉 GGUF。每组将相同的三路 256×256 RGB、指令、六维已归一化状态和 50×32 固定噪声输入两种实现，比较 50×6 的最终归一化动作。没有应用状态归一化或动作反归一化，结果不表示机器人实际单位的误差或任务成功率。

用例以固定种子 `20261030` 合成：`baseline` 为随机图像、状态与噪声；`instruction`、`state`、`camera`、`noise` 分别只改变指令、状态、第二路图像或噪声；`black_long_instruction` 使用三路黑图、零状态和会触发左侧截断的长指令。每对输入的噪声完全相同。四个 CPU 线程；LeRobot/PyTorch 为本机环境，Edge 服务在所有用例中保持运行；时长不应作为性能基准。

| 用例 | 最大绝对误差 | 平均绝对误差 | RMSE | 相对 L2 | 余弦相似度 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.015718 | 0.004820 | 0.006323 | 1.3847% | 0.999915783 |
| instruction | 0.021160 | 0.002774 | 0.004060 | 0.8858% | 0.999965578 |
| state | 0.020618 | 0.003982 | 0.005004 | 1.0152% | 0.999954877 |
| camera | 0.009701 | 0.003124 | 0.003828 | 0.8361% | 0.999965106 |
| noise | 0.016838 | 0.005107 | 0.006594 | 1.3887% | 0.999916626 |
| black_long_instruction | 0.010493 | 0.001811 | 0.002540 | 0.3024% | 0.999995427 |

相对 L2 定义为 `||Edge - LeRobot||₂ / ||LeRobot||₂`，其余误差在 300 个动作数值上计算。各动作维度的平均绝对误差：

| 用例 | 维度 0 | 维度 1 | 维度 2 | 维度 3 | 维度 4 | 维度 5 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.002603 | 0.007693 | 0.001610 | 0.010076 | 0.006016 | 0.000925 |
| instruction | 0.002611 | 0.003417 | 0.002597 | 0.002818 | 0.001713 | 0.003486 |
| state | 0.004001 | 0.002891 | 0.001484 | 0.005344 | 0.004508 | 0.005664 |
| camera | 0.001714 | 0.002780 | 0.004499 | 0.002953 | 0.005582 | 0.001218 |
| noise | 0.004024 | 0.007072 | 0.002349 | 0.010106 | 0.006258 | 0.000835 |
| black_long_instruction | 0.000930 | 0.001238 | 0.001556 | 0.000935 | 0.004643 | 0.001566 |

## 复现

```bash
cmake --build build --target llama-server
HF_HUB_OFFLINE=1 python tools/smolvla-reference/compare_http_cases.py \
  --checkpoint ../models/smolvla_base \
  --backbone /path/to/SmolVLM2-500M-Video-Instruct \
  --model tmp/smolvla-session/smolvla-main.gguf \
  --mmproj tmp/smolvla-session/mmproj-smolvla.gguf \
  --server build/bin/llama-server \
  --output tmp/smolvla-session/http-comparison
```

脚本参照 README 的 foreground 方式启动本地 CPU 服务，逐组调用 `/foreground/reset`、`/foreground/image`、`/foreground/state` 和 `/foreground/infer`。固定噪声文件仅在同步请求之间更新，确保每组两种实现的初始噪声相同。输出目录保存 PNG 图像、状态、噪声、HTTP 响应和服务日志；`actions.npz` 包含两种实现的全部动作，`report.json` 包含逐维误差和启动命令。服务使用说明见 [SmolVLA 适配文档](smolvla-adaptation-status.md)。
