# Qwen3-TTS-ROCm

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20ROCm%207.14.0-orange.svg)](https://rocm.docs.amd.com/)
[![Hardware](https://img.shields.io/badge/hardware-gfx1151-red.svg)](https://rocm.docs.amd.com/)

[English](README.md) | **简体中文**

> 本文件是 [`README.md`](README.md) 的逐节中文对照版本：章节一一对应、表格与数字完全一致；
> 所有代码块原文保留。锚点在两个文件中使用相同的显式 HTML id。

让官方 Qwen3-TTS 语音合成在 AMD Ryzen AI Max+ PRO 395 / Radeon 8060S iGPU（`gfx1151`）上
**零补丁**运行。一条命令安装 AMD 锁定版本的 ROCm 7.14.0 PyTorch 轮子，另一条下载
六个官方仓库权重，第三条启动增强版双语五标签页 Gradio 演示（`http://localhost:8000`）。
所有合成调用都停留在未修改的官方 API 上——我们的 loader 直接交还原生模型对象，
并且有一条上游一致性测试为此作证。

## 为什么有这个项目

Qwen3-TTS 以 CUDA 优先的方式发布：上游假定 NVIDIA GPU 与 flash-attn 内核库，而
`gfx1151` 一类的集成显卡开箱即不可能工作。本仓库刻意**不是**那些代码的 fork——它是围绕
**未经修改的官方 `qwen-tts` 包**的一层薄壳：环境诊断、智能默认的模型加载器、双源
（ModelScope / hf-mirror）下载器和一个增强版演示界面，仅此而已。核心承诺见下文的
[零修改保证](#zero-modification-guarantee)，并由专门的一致性测试强制执行；如果信任这个
仓库之前只读一节，那就读那一节。

<a id="zero-modification-guarantee"></a>

## 零修改保证

* [`loader.load()`](src/qwen3_tts_rocm/loader.py) 返回的就是**官方
  `qwen_tts.Qwen3TTSModel.from_pretrained` 的返回值本身**——原生模型对象，绝无包装。
  智能默认值（HIP GPU 上 `bfloat16` + `sdpa`）只通过公开的官方关键字参数施加。
* 证据就在测试套件里：
  [`tests/test_official_demo_parity.py`](tests/test_official_demo_parity.py)
  用*我们*的 loader 加载出的模型对象构建原封不动的上游
  `qwen_tts.cli.demo.build_demo()`，再经 Gradio 自身的事件注册表执行演示自己的回调闭包
  ——包括一次真实的 GPU 合成。绿色通过的运行记录存档于
  [`evidence/official-parity.txt`](evidence/official-parity.txt)。
* 项目政策（见 [`CONTRIBUTING.md`](CONTRIBUTING.md) 与
  [`NOTICE`](NOTICE)）：永远不内置、不补丁上游源码；`pyproject.toml`
  原样依赖已发布的 `qwen-tts==0.1.1` 制品。

## 环境要求

| 组件 | 要求 |
|---|---|
| APU / iGPU | AMD Ryzen AI Max+ PRO 395（搭载 Radeon 8060S Graphics）—— 架构 `gfx1151`（Strix Halo 级）；开发与验证均在此设备上完成 |
| 内存 | 统一内存级硬件：CPU 与 iGPU 共享 94 GB LPDDR5X 池（torch/HIP 可见约 80 GiB） |
| 内核 | Linux 且 iGPU 由 `amdgpu` DRM 驱动接管（用 `rocm-smi` 验证）；已在 `6.17.0-1032-oem` 内核上验证 |
| ROCm | 来自 AMD pip 索引 `https://repo.amd.com/rocm/whl-multi-arch/` 的 ROCm 7.14.0 代际轮子：`torch[device-gfx1151]==2.12.0+rocm7.14.0`、`torchvision[device-gfx1151]==0.27.0+rocm7.14.0`、`torchaudio==2.11.0+rocm7.14.0` —— 由 `scripts/install.sh` 自动安装，无需手敲 |
| Python | ≥ 3.10（在 3.12 上验证） |
| 磁盘 | 约 18 GB 空闲空间，存放六个官方仓库（各自内含语音分词器副本；权重位于 `models/` 下且从不入库提交） |
| 网络 | 可达 ModelScope（`modelscope.cn`）——默认通道在 CN 网络内直接可用；当 `huggingface.co` 被阻断时回退传输自动改走 `hf-mirror.com`，因此无需 VPN |

## 快速开始

从零到会说话的浏览器标签页只需四条命令：

```bash
git clone https://github.com/<OWNER>/Qwen3-TTS-ROCm.git   # placeholder — replace <OWNER> after push
cd Qwen3-TTS-ROCm
bash scripts/install.sh          # venv + pinned AMD ROCm wheels + editable install + GPU gate
bash scripts/download_models.sh  # six official checkpoints, ModelScope-first, hf-mirror fallback
bash scripts/run_demo.sh         # enhanced demo -> http://localhost:8000
```

*发布前请将 `<OWNER>` 替换为你的 GitHub 用户名。*

说明：

* `scripts/install.sh` 幂等（可安全重复执行），结尾是一次 GPU 健全性闸门，成功时打印
  `SPIKE-GPU-OK`。
* 模型权重下载到 `<repo>/models/`（可用 `$QWEN3_TTS_ROCM_MODELS_DIR` 改写）。中断可续传；
  已完成的仓库会被短路跳过，因此重跑代价很低。
* `scripts/install.sh --with-models` 会串联下载步骤。
* 安装完成后，`qwen3-tts-rocm-check` 随时可做双语环境自检（只读，绝不抛异常）。

### 最小 Python 示例

最能证明 loader 承诺的最小程序——只经过我们加载一次，此后一切都是纯官方 API，
与上游 README 快速开始一致：

```python
from qwen3_tts_rocm import loader

tts = loader.load("custom-voice")            # sdpa/bf16 defaults on gfx1151
wavs, sr = tts.generate_custom_voice(text="你好，ROCm。", language="auto",
                                     speaker=tts.get_supported_speakers()[0])
```

* `loader.load("custom-voice")` 把注册表别名解析到模型目录下的
  `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` 并应用 ROCm 智能默认值
  （`device_map=auto`、`dtype=bfloat16`、HIP 上 `attn_implementation=sdpa`）。
  返回的 `tts` 就是原生官方对象；`generate_custom_voice`、
  `get_supported_speakers()`……都是原封不动的官方方法。
* 语言取值由官方 getter 返回时一律为**小写**
  （`"auto"`、`"chinese"`、`"english"`……）——调用 API 时请传小写。
  展示大小写（"Chinese"）由演示 UI 层处理。
* 加载永远不会在背后偷偷下载多 GB 权重：目标缺失时会抛出 `RuntimeError`，
  并指明 `bash scripts/download_models.sh` 是补救办法。

## 五标签页 Gradio 演示

`bash scripts/run_demo.sh` 提供增强版双语（中文/English）演示应用，包含**五个标签页**，
由带服务端文件校验的无界面 `SynthesisService` 支撑：

1. **语音克隆（Voice Clone）** —— 参考音频克隆，含保存/加载音色子标签页，
   可持久化为可复用的 `.pt` 提示文件。
2. **预设音色（Preset Speakers）** —— 任选内置说话人并附加可选指令，即刻合成。
3. **音色设计（Voice Design）** —— 用自然语言描述想要的音色并生成。
4. **编解码器（Codec）** —— 经官方 12Hz 语音分词器的编码→解码往返可视化，
   附码率/步数元信息与可下载 WAV。
5. **合成历史（History）** —— 试听、下载或删除当前会话生成的片段。

侧边栏承载模型切换器（同一时刻只有**一个** TTS 模型驻留——LRU 槽位为一）、实时
VRAM/GTT 状态行以及高级采样参数折叠区（留空即默认值）。

### 排队串行说明

启动脚本默认传入 `--concurrency 1`（上游默认队列并发 16 在此毫无意义）：生成请求严格
逐个执行，因为单 GPU 统一内存设备同一时刻只驻留一个模型，并发合成本来也会被串行化进
同一块算力池。同样的说明也会显示在演示页脚。除非清楚缘由，否则不要调高它。

### 麦克风采集需要 HTTPS（或 localhost）

依照上游所依赖的浏览器安全模型，Voice Clone / Codec 标签页中的麦克风输入需要安全上下文：
要么在本机打开页面（`http://localhost:8000`），要么通过 TLS 提供演示。改编自上游说明，
一行生成自签名证书对：

```bash
openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 365 \
    -out cert.pem -subj "/CN=localhost"
bash scripts/run_demo.sh --ssl-certfile cert.pem --ssl-keyfile key.pem
```

接受浏览器关于自签名证书的一次性警告后，局域网其他设备也能通过 HTTPS 使用麦克风采集。

### 截图

以下截图捕获自 v0.1.0 验收阶段在 gfx1151 真机上的运行
（双语界面，经 Gradio 队列完成真实合成）：

| | |
|---|---|
| ![Voice Clone 语音克隆](docs/img/demo-clone.png) | ![Preset Speakers 预设音色](docs/img/demo-customvoice.png) |
| ![Voice Design 音色设计](docs/img/demo-voicedesign.png) | ![Codec 编解码器](docs/img/demo-codec.png) |
| ![History 合成历史](docs/img/demo-history.png) | *语音克隆 · 预设音色 · 音色设计 · 编解码器 · 合成历史* |

`demo-voicedesign.png` 与 `demo-history.png` 中可见侧边栏模型切换器的会话
状态：`[voice-design] loaded (已驻留)` 及实时显存读数、一段 5.6 秒的合成
结果，以及按时间倒序、支持试听/下载/删除的历史列表。

## 性能速览（gfx1151）

实测中位实时率（RTF：每生成一秒音频消耗的实际秒数，越低越好）——Ryzen AI Max+ PRO 395、
bfloat16/sdpa、短句与中等长度文本、上限 `max_new_tokens=512`：

| 模型族 | 中位 RTF 区间 |
|---|---|
| 定制音色，`custom-voice` | 1.31 – 1.51 |
| 定制音色，`voice-design` | 1.27 – 1.62 |
| 零样本声音克隆（`base`） | 1.71 – 1.88 |

也就是说，在 iGPU 上等几秒得到几秒语音——句子级交互演示可用，批量离线合成更是绰绰有余。
我们只给区间，不做绝对延迟承诺：在统一内存共享池上，数字随时钟、温度、内存压力与后台
负载漂移，任何单次结果都只能视作指示性（完整分格表格、方法学、n=2 注意事项与复现命令见
[`docs/benchmarks.md`](docs/benchmarks.md)）。

> **退化生成警告：** 在上游默认的 `max_new_tokens=2048` 下，采样偶尔会陷入退化循环，
> 在 iGPU 上连续渲染数分钟（实测一次失控约 23 分钟）。因此本仓库的所有服务默认启用
> **512 token 护栏**——确有必要时再通过演示的高级折叠区或 `gen_kwargs` 有意识地覆盖。

## FAQ 与故障排查

症状 → 修复位置。配套指南展开本项目的诊断输出的每一条 `ERROR:` / `WARN:` / `INFO:`。

| 症状或消息 | 修复文档 |
|---|---|
| `PyTorch is not installed or not importable` | [docs/troubleshooting.md](docs/troubleshooting.md) |
| `The installed PyTorch is NOT an AMD ROCm/HIP build` | [docs/troubleshooting.md](docs/troubleshooting.md) |
| `torch.cuda.is_available() is False` / `/dev/kfd` 权限投诉 | [docs/troubleshooting.md](docs/troubleshooting.md) |
| `HSA_OVERRIDE_GFX_VERSION` 警告 | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Hugging Face 与 ModelScope 双双下载失败 | [docs/troubleshooting.md](docs/troubleshooting.md) |
| 统一内存上的显存不足或超长生成 | [docs/troubleshooting.md](docs/troubleshooting.md) |
| 请求了 `flash_attention_2` 但缺 flash-attn | [docs/troubleshooting.md](docs/troubleshooting.md) |
| SoX 横幅 / MIOpen 控制台刷屏 | [docs/troubleshooting.md](docs/troubleshooting.md) |

两个最常见问题的快速回答：

* *“这是一个模型吗？”* 不是——模型留在官方仓库里单独下载（合计约 18 GB）。
  本项目只是包裹它们的一层胶水。
* *“需要把权重提交进 git 吗？”* 永远不需要；参见 `.gitignore` 与
  `$QWEN3_TTS_ROCM_MODELS_DIR`。

## 归属与免责声明

* Qwen3-TTS 及全部模型权重出自 **阿里巴巴 Qwen 团队**的工作
  （[上游仓库](https://github.com/QwenLM/Qwen3-TTS)，Apache-2.0；权重遵循阿里自有的
  Qwen 模型许可——下载即接受其条款，详见 [`NOTICE`](NOTICE)）。本项目不再分发模型权重。
* Qwen3-TTS-ROCm 是一个**非官方社区适配版**；与阿里巴巴及 AMD 无隶属、背书或出品关系。
* 上游音频生成免责声明简版（自官方演示页脚节选浓缩）：
  *生成的音频可能不准确或不恰当，不代表任何人的立场；如何合法使用由你自行负责——
  禁止制造违法、有害、深度伪造或侵权内容。*

  中文（同义简版，自上游页脚节选）：*音频由 AI 模型自动生成，可能不准确或不当，
  不代表任何一方立场；请依法使用，严禁生成违法、有害、深度伪造或侵权内容。*

## 参与贡献与联系

* 开发环境搭建、守则与 PR 清单：[`CONTRIBUTING.md`](CONTRIBUTING.md)。头条规则：
  贡献必须维护上文“薄壳保证”——不改上游源码、不入库权重。
* Bug 反馈与功能建议请到本仓库的 issue 跟踪器提交。
* 本项目的主页：`https://github.com/<OWNER>/Qwen3-TTS-ROCm` —— `<OWNER>` 是维护者账号的
  占位常量，只在此定义一次并被上方 Quickstart 克隆地址引用；推送后两处一并更新。

## 许可证

代码：Apache-2.0 —— 见 [`LICENSE`](LICENSE)。模型权重仍受阿里自有模型许可约束
（出处与下载条款见 [`NOTICE`](NOTICE)）。
