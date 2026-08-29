# Qwen3-TTS on AMD ROCm

**官方 Qwen3-TTS。AMD Radeon。零上游补丁。**

在 AMD Ryzen AI Max+ PRO 395 / Radeon 8060S（`gfx1151`）上原样运行官方
[`qwen-tts`](https://github.com/QwenLM/Qwen3-TTS) 包：一条命令安装 AMD 锁定
版本的 ROCm 7.14.0 PyTorch 轮子，一条下载官方权重，一条启动双语五标签页
Gradio 演示（`http://localhost:8000`）。所有合成调用全部走未经修改的官方 API。

[English](README.md) | **简体中文**

[![CI](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AIwork4me/Qwen3-TTS-ROCm/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/AIwork4me/Qwen3-TTS-ROCm)](https://github.com/AIwork4me/Qwen3-TTS-ROCm/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/)
[![ROCm](https://img.shields.io/badge/ROCm-7.14.0-orange.svg)](https://rocm.docs.amd.com/)
[![Hardware](https://img.shields.io/badge/hardware-gfx1151%20%7C%20Radeon%208060S-red.svg)](docs/benchmarks.md#platform)

非官方社区项目 —— 与阿里巴巴及 AMD 无隶属或背书关系，见[归属与免责声明](#归属与免责声明)。

## 验证结果，而非承诺

| 验证项 | 结果 |
|---|---|
| 官方模型仓库 | **6 / 6 已验证** —— 5 个 TTS checkpoint + tokenizer |
| 自动化测试 | **232 通过** —— 207 CPU + 25 真机 GPU；另 1 项 HIP 相关跳过 |
| 对上游 `qwen-tts` 的补丁 | **0** —— 由专门的一致性测试强制保证 |
| GPU · ROCm | Radeon 8060S（`gfx1151`）· ROCm 7.14.0（`torch 2.12.0+rocm7.14.0`） |
| 精度 / 注意力 | bfloat16 · PyTorch SDPA —— 本次验证栈未启用 FlashAttention |
| 证据 | 逐字运行记录见 [`evidence/`](evidence/README.md) |

共收集 233 项测试；在 AMD ROCm 主机上该 HIP 门控的 CPU 检查会照常执行，
因此验证主机本身为 233/233 全通过（208 CPU + 25 GPU）。CPU 测试套件在
CI 中于 Python 3.10 / 3.11 / 3.12 上运行。

下面是验证真机上实拍的演示标签页：

![Preset Speakers 标签页在 Radeon 8060S 上合成](docs/img/demo-customvoice.png)

🔊 **听一下** —— [5.5 秒中文示例](evidence/demo-rest-gen-zh.wav)，
由验证运行中经演示 REST API 生成，过程记录见
[`evidence/demo-smoke.txt`](evidence/demo-smoke.txt)。

## 快速开始

### 🚀 先跑起来（约 5 GB 下载）

入门**不需要**一次下齐六个权重。`tokenizer` + `custom-voice` 子集（约 5 GB）
足以运行下面的 Python 片段，以及演示中的**预设音色**和**编解码器**标签页：

```bash
git clone https://github.com/AIwork4me/Qwen3-TTS-ROCm.git
cd Qwen3-TTS-ROCm
bash scripts/install.sh                              # venv + AMD 锁定 ROCm 轮子 + GPU 闸门
bash scripts/download_models.sh tokenizer custom-voice   # 约 5 GB，ModelScope 优先
bash scripts/run_demo.sh                             # -> http://localhost:8000
```

也可以跳过界面——最能证明 loader 承诺的最小程序（只经我们加载一次，
之后全是纯官方 API，与上游快速开始一致）：

```python
from qwen3_tts_rocm import loader

tts = loader.load("custom-voice")            # sdpa/bf16 defaults on gfx1151
wavs, sr = tts.generate_custom_voice(text="你好，ROCm。", language="auto",
                                     speaker=tts.get_supported_speakers()[0])

import soundfile as sf
sf.write("hello-rocm.wav", wavs[0], sr)  # 保存 / save
```

更轻的选择：`bash scripts/download_models.sh tokenizer custom-voice-0.6b`
（约 3 GB），片段中改用 `custom-voice-0.6b` 别名即可。

**预期表现：** 模型加载时长与文件系统缓存状态强相关——验证真机上同会话内
加载为 2.2–4.9 秒，冷启动更久；首次运行还有一次性的 MIOpen 内核调优
（之后有缓存）。终端只打印一行 loader 状态提示；首见的 MIOpen 控制台刷屏
属正常（[故障排查](docs/troubleshooting.md)）。权重下载到 `models/`
（可用 `$QWEN3_TTS_ROCM_MODELS_DIR` 改写）；中断可续传，重跑会跳过已完成的仓库。

### 完整体验（约 18 GB）

```bash
bash scripts/download_models.sh   # 全部六个官方仓库
bash scripts/run_demo.sh
```

| 别名 | 官方仓库 | 大小 |
|---|---|---|
| `tokenizer` | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | 651M |
| `custom-voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 4.3G |
| `voice-design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 4.3G |
| `base` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 4.3G |
| `custom-voice-0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 2.4G |
| `base-0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 2.4G |

模型不齐时演示照常启动；缺哪一族模型，对应标签页会明确提示而不是莫名报错。
`scripts/install.sh --with-models` 可串联安装与下载；`qwen3-tts-rocm-check`
随时可做双语环境自检（只读，绝不抛异常）。

## 你能得到什么

* **`qwen3-tts-rocm-check`** —— 双语 ROCm 环境自检，绝不抛异常。
* **`loader.load(alias)` / `loader.unload()`** —— 通过官方公开参数施加智能
  默认值（HIP 上 `device_map=auto`、`bfloat16`、`sdpa`），返回**原生官方模型
  对象**，绝无包装。加载绝不会在背后偷偷下载权重——模型缺失时抛出
  `RuntimeError` 并直接给出下载命令。
* **六别名下载器** —— ModelScope 优先、`hf-mirror.com` 回退（记录中的验证
  主机即在中国大陆网络环境下全程免 VPN 完成下载；其他网络环境可能有差异），
  支持断点续传，可按别名或整批下载。
* **五标签页双语 Gradio 演示**（中文/English）：① 语音克隆（含可保存/复用的
  音色提示）· ② 预设音色 · ③ 音色设计 · ④ 编解码器往返 · ⑤ 合成历史。
  侧边栏含模型切换器（同一时刻只驻留一个模型）、实时 VRAM/GTT 读数与高级
  采样参数折叠区。
  <details>
  <summary>五个标签页截图</summary>

  | | |
  |---|---|
  | ![Voice Clone 语音克隆](docs/img/demo-clone.png) | ![Preset Speakers 预设音色](docs/img/demo-customvoice.png) |
  | ![Voice Design 音色设计](docs/img/demo-voicedesign.png) | ![Codec 编解码器](docs/img/demo-codec.png) |
  | ![History 合成历史](docs/img/demo-history.png) | *语音克隆 · 预设音色 · 音色设计 · 编解码器 · 合成历史* |

  </details>

  演示说明：生成按队列并发 1 串行执行——单 GPU 统一内存设备只驻留一个模型，
  并发合成本来也会被串行化进同一块算力池。克隆/编解码器标签页的麦克风输入
  需要安全上下文：在本机用 `http://localhost:8000` 打开，或改走 TLS（先用一行
  `openssl` 生成自签名证书对，再 `bash scripts/run_demo.sh --ssl-certfile
  cert.pem --ssl-keyfile key.pem`）。详见
  [`docs/troubleshooting.md`](docs/troubleshooting.md)。
* **512 token 生成护栏** —— 限定渲染时长；上游默认 2048 下采样偶尔会退化成
  数分钟的循环（实测一次约 23 分钟）。确有必要时再经高级折叠区或
  `gen_kwargs` 覆盖。
* **可复现的 RTF 基准**（`scripts/benchmark.py`），方法学公开、原始输出存档。
* **Docker 镜像**，含 `/dev/kfd` + `/dev/dri` 直通
  （[docker/README.md](docker/README.md)）。
* **测试套件** —— 232 项通过（207 CPU + 25 真机 GPU 集成；共收集 233 项，
  仅 CPU 的 CI 上有 1 项 HIP 相关跳过），含下文的上游一致性证明。

<a id="why-this-project-exists"></a>

## 为什么选择 Qwen3-TTS-ROCm？

上游 Qwen3-TTS 的部署文档以 CUDA / FlashAttention 为主。本项目在不修改官方
`qwen-tts` 包的前提下，补充一条经过验证的 `gfx1151` ROCm 部署路径——只是一层
薄壳：环境诊断、智能默认加载器、双源下载器和增强版演示界面。不是 fork；
永远不内置、不补丁上游源码。

| 能力 | 上游 Qwen3-TTS | Qwen3-TTS-ROCm |
|---|---|---|
| 官方 Qwen3-TTS API | ✅ | ✅（未修改） |
| 官方模型权重 | ✅ | ✅（不再分发） |
| gfx1151 ROCm 路径已验证 | — | ✅ |
| 一条命令安装 ROCm 轮子 | — | ✅ |
| ROCm 环境自检 | — | ✅ |
| ModelScope 优先下载器 | — | ✅ |
| AMD iGPU 上的 RTF 基准证据 | — | ✅ |

<a id="兼容性"></a>

## 兼容性

| GPU / 平台 | 架构 | ROCm | 状态 | 证据 |
|---|---|---|---|---|
| Radeon 8060S / Ryzen AI Max+ PRO 395 | `gfx1151` | 7.14.0 | ✅ 已验证 —— 唯一经过独立验证的配置 | [`evidence/`](evidence/README.md) |
| 其他 ROCm capable AMD GPU | — | — | 🧪 **尚未验证 —— 欢迎社区实测** | 提交 issue 并附上 `qwen3-tts-rocm-check` 输出 |

loader 的 HIP 默认值是通用的，但本仓库的每个数字与结论都只追溯到上表这一
个已验证配置。请勿臆断其他显卡能或不能用——非常欢迎其他 ROCm 硬件的实测
反馈，验证后会在表中列出。注意：仓库自带 `scripts/install.sh` 是已验证的
`gfx1151` 安装路径（锁定 `device-gfx1151` 轮子）；测试其他架构时，请使用
对应的 ROCm PyTorch 栈，并在验证报告中记录完整安装方式。

**在其他 AMD GPU 上跑通了？[提交硬件验证报告](https://github.com/AIwork4me/Qwen3-TTS-ROCm/issues/new?template=hardware-validation.yml)**——只收实测结果，验证后兼容性矩阵随你扩展。

## 性能

在 Ryzen AI Max+ PRO 395、bfloat16/sdpa、短句与中等长度文本、上限
`max_new_tokens=512` 条件下实测的**中位 RTF**——*每生成一秒音频消耗的实际
秒数，越低越好*。RTF 1.3 的含义是：生成 1 秒音频约消耗 1.3 秒计算时间。

| 工作负载 | 中位 RTF |
|---|---:|
| 定制音色 Custom Voice（1.7B） | 1.31 – 1.51 |
| 音色设计 Voice Design（1.7B） | 1.27 – 1.62 |
| 零样本语音克隆（`base` 1.7B） | 1.71 – 1.88 |

在 iGPU 上等几秒得到几秒语音——句子级交互演示可用，批量离线合成更是从容。
我们只给区间，不做绝对延迟承诺：统一内存共享池上，数字随时钟、温度、内存
压力与后台负载漂移。完整分格表格、方法学、n=2 注意事项与复现命令见
[`docs/benchmarks.md`](docs/benchmarks.md)。

## 验证与可复现性

<a id="zero-modification-guarantee"></a>

* [`loader.load()`](src/qwen3_tts_rocm/loader.py) 返回的就是**官方
  `qwen_tts.Qwen3TTSModel.from_pretrained` 的返回值本身**——原生模型对象，
  绝无包装；智能默认值只通过公开的官方关键字参数施加。
* 证明靠测试：[`tests/test_official_demo_parity.py`](tests/test_official_demo_parity.py)
  用*我们* loader 加载出的模型对象构建原封不动的上游
  `qwen_tts.cli.demo.build_demo()`，再经 Gradio 自身的事件注册表执行演示
  自己的回调闭包——含一次真实 GPU 合成。绿色通过记录：
  [`evidence/official-parity.txt`](evidence/official-parity.txt)。
* 项目政策（见 [`CONTRIBUTING.md`](CONTRIBUTING.md) 与 [`NOTICE`](NOTICE)）：
  不内置、不补丁上游源码；`pyproject.toml` 原样依赖已发布的
  `qwen-tts==0.1.1` 制品。

自己动手复核：

```bash
bash scripts/verify_gpu.sh                    # ROCm 正常时打印 SPIKE-GPU-OK
qwen3-tts-rocm-check                          # 环境自检
python -m pytest -m "not gpu and not requires_download" -q   # 207 个 CPU 测试（AMD 主机 208 个）
python -m pytest -m "gpu" -q                  # 25 个 GPU 测试（需权重）
.venv/bin/python scripts/benchmark.py         # 全新 RTF 数据
```

每个引用的数字都可追溯到 [`evidence/README.md`](evidence/README.md) 中列出
的逐字记录。

## 已验证配置

开发与验证均在这台机器上完成（这是"实测配置"，不是"最低要求"）：

| 事实 | 数值 |
|---|---|
| APU | AMD Ryzen AI Max+ PRO 395（Radeon 8060S，`gfx1151`，Strix Halo 级） |
| 内存 | 94 GB LPDDR5X 统一内存池，torch/HIP 可见约 80 GiB |
| 内核 | Linux 6.17.0-1032-oem，`amdgpu` DRM 驱动（用 `rocm-smi` 确认） |
| ROCm / torch | 来自 `repo.amd.com` 的 7.14.0 代际轮子：`torch[device-gfx1151]==2.12.0+rocm7.14.0`（含 torchvision/torchaudio）—— 由 `scripts/install.sh` 自动安装，无需手敲 |
| Python | 验证主机为 3.12；CPU CI 矩阵运行 3.10 / 3.11 / 3.12 |

### 需求与已知约束

* **磁盘** —— 最小子集约 5 GB，六个仓库全量约 18 GB（权重位于 `models/`，
  从不提交入库）。
* **网络** —— 需可达 ModelScope（`modelscope.cn`）；`huggingface.co` 被阻断
  时回退传输自动改走 `hf-mirror.com`。记录中的验证主机在中国大陆网络下全程
  免 VPN 完成了全部下载——其他网络环境可能有差异。
* **GPU** —— 仅在 `gfx1151` 上验证过（见[兼容性](#兼容性)）；需要 `amdgpu`
  DRM 驱动正常工作，且能访问 `/dev/kfd` + `/dev/dri`（`render`/`video` 组）。
* **内存** —— 我们会话中驻留的 1.7B 模型（bf16）占用统一内存池约 4.6 GiB；
  更小内存机器的最低要求**未经实测**。共享统一内存池上，关闭吃内存的桌面
  应用可获得更好 RTF。

## Docker

可复现的 Docker 构建，通过 `/dev/kfd` + `/dev/dri` 直通在 ROCm 上运行；把
宿主机的 `models/` 目录挂载进镜像声明的卷即可：

```bash
docker build -f docker/Dockerfile -t qwen3-tts-rocm:dev .
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    -v "$PWD/models:/workspace/models" \
    -p 8000:8000 \
    qwen3-tts-rocm:dev
```

镜像验证记录：[`evidence/docker-build-final.txt`](evidence/docker-build-final.txt)。
完整指南——组 GID 注意事项、无 GPU 诊断、无 GPU 冒烟测试——见
[`docker/README.md`](docker/README.md)。

## 故障排查

先运行 `qwen3-tts-rocm-check`，再到
[`docs/troubleshooting.md`](docs/troubleshooting.md) 对症查找——该指南逐条
展开诊断输出的每一条 `ERROR:` / `WARN:` / `INFO:`：

| 症状 | 修复文档 |
|---|---|
| `torch.cuda.is_available() is False` / `/dev/kfd` 权限 | [docs/troubleshooting.md](docs/troubleshooting.md) |
| `The installed PyTorch is NOT an AMD ROCm/HIP build` | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Hugging Face 与 ModelScope 双双下载失败 | [docs/troubleshooting.md](docs/troubleshooting.md) |
| 显存不足或超长生成 | [docs/troubleshooting.md](docs/troubleshooting.md) |
| 首次运行的 MIOpen / SoX 控制台刷屏 | [docs/troubleshooting.md](docs/troubleshooting.md) |

两个常见问题的快速回答：**"这是一个模型吗？"** 不是——模型留在官方仓库
单独下载，本项目只是包裹它们的胶水。**"权重会提交进 git 吗？"** 永远不会；
参见 `.gitignore` 与 `$QWEN3_TTS_ROCM_MODELS_DIR`。

## 参与贡献

守则、开发环境搭建与 PR 清单见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。头条
规则：贡献必须维护零修改保证——不改上游源码、不入库权重。Bug 反馈与硬件
实测请到 [issue 跟踪器](https://github.com/AIwork4me/Qwen3-TTS-ROCm/issues)。

<a id="归属与免责声明"></a>

## 归属与免责声明

* Qwen3-TTS 及全部模型权重出自**阿里巴巴 Qwen 团队**的工作
  （[上游仓库](https://github.com/QwenLM/Qwen3-TTS)，Apache-2.0；权重遵循
  阿里自有的 Qwen 模型许可——下载即接受其条款，详见 [`NOTICE`](NOTICE)）。
  本项目不再分发模型权重。
* Qwen3-TTS-ROCm 是一个**非官方社区适配版**；与阿里巴巴及 AMD 无隶属、
  背书或出品关系。
* 上游演示页脚节选浓缩：*生成的音频可能不准确或不当，不代表任何人的立场；
  如何合法使用由你自行负责——禁止制造违法、有害、深度伪造或侵权内容*
  （English: *generated audio may be inaccurate or inappropriate, does not
  represent anyone's views, and it is your responsibility to use it
  lawfully*）。

## 许可证

代码：Apache-2.0 —— 见 [`LICENSE`](LICENSE)。模型权重仍受阿里自有模型许可
约束（出处与下载条款见 [`NOTICE`](NOTICE)）。
