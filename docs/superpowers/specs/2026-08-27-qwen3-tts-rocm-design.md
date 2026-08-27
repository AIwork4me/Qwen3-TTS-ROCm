# Qwen3-TTS-ROCm 设计文档

- 日期：2026-08-27
- 状态：已获用户批准（整体批准，7 节）
- 路线：薄适配层 shim（官方包零改动 + 我们的适配层）；若 Spike 发现必须改 core 则升级为 vendor+patches（用户已同意该升级预案）

## 1. 目标与非目标

### 目标
1. 让 https://github.com/QwenLM/Qwen3-TTS 的 Python 包 `qwen-tts==0.1.1` 在本机 AMD Radeon 8060S（gfx1151，Ryzen AI Max+ PRO 395）上基于 **ROCm 7.14.0** 完整运行。
2. **支持官方 Python Package 的全部公开功能**：
   - `Qwen3TTSModel.from_pretrained / generate_custom_voice / generate_voice_design / generate_voice_clone / create_voice_clone_prompt / get_supported_speakers / get_supported_languages`
   - `VoiceClonePromptItem` 数据类与克隆提示保存/复用（含 x-vector only 模式）
   - `Qwen3TTSTokenizer.from_pretrained / encode / decode / get_model_type / get_input_sample_rate / get_output_sample_rate / get_encode_downsample_rate / get_decode_upsample_rate`
   - 批量推理、全部生成参数透传（max_new_tokens、temperature、top_k/top_p、repetition_penalty、subtalker_*）
3. 复刻 ModelScope Demo（实为官方包内 Gradio 应用）并以**增强版**形态提供图形界面：模型懒加载切换、合成历史试听/下载、示例参考音频、中英双语 UI、编解码器可视化页签。
4. 发布顶级开源品质项目：双语 README、标准仓库结构、Apache-2.0、一键安装脚本、pytest 测试套件、Docker 支持、PyPI 发布 `qwen3-tts-rocm`。

### 非目标（YAGNI）
- 不做模型微调训练流水线（官方亦未在本仓库提供）
- 不做 vLLM-Omni 服务化部署
- 不做真流式音频输出 API（官方包 v0.1.1 无此接口；Demo 一次性返回完整音频）
- 不支持 Windows/macOS；Linux + ROCm 7.14.0 pip wheel 为唯一官方支持路径
- 不改动/不分发上游源码，不上游 PR

## 2. 平台与环境事实（已验证）

| 项 | 值 |
|---|---|
| GPU | AMD Radeon 8060S，gfx1151，40 CU（Ryzen AI Max+ PRO 395 APU，统一内存） |
| 内存 | 94GB 统一内存，可用 ~89GB |
| 内核 | 6.17.0-1032-oem（amdgpu DRM 正常，rocm-smi 工作正常） |
| 系统 ROCm 用户态 | /opt/rocm → 7.2.1（仅系统自带；torch 走 pip wheel 自带运行时，二者共存无冲突是前提之一，Spike 验证） |
| Python | 系统 3.12.3；工具链有 uv（~/.local/bin/uv）、pip、pipx |
| 磁盘 | / 空闲 1.2TB |
| 网络 | PyPI ✅、repo.amd.com ✅、modelscope.cn ✅、hf-mirror.com ✅；github.com 与 huggingface.co ❌（代码经 PyPI sdist 获取成功；模型走 ModelScope） |
| Wheel 验证 | repo.amd.com 上已确认存在 `torch-2.12.0+rocm7.14.0-cp312 linux_x86_64.whl` 与 `torchaudio-2.11.0+rocm7.14.0` |

**torch 安装命令（AMD 官方，作为 install.sh 核心）**：

```bash
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "torch[device-gfx1151]==2.12.0+rocm7.14.0" \
    "torchvision[device-gfx1151]==0.27.0+rocm7.14.0" \
    "torchaudio==2.11.0+rocm7.14.0"
```

注意安装顺序：必须先从 AMD index 装 torch 三件套，再装其他 PyPI 依赖，防止拉到 PyPI 默认 CUDA 版 torch。

## 3. 模型仓（6 个，全部覆盖）

| 别名 | ModelScope/HF 仓库 | 用途 |
|---|---|---|
| tokenizer | Qwen/Qwen3-TTS-Tokenizer-12Hz | 编解码器（12Hz v2 路径） |
| voice-design | Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | 音色设计（含指令控制）|
| custom-voice | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | 9 预设音色 + 指令控制 |
| base | Qwen/Qwen3-TTS-12Hz-1.7B-Base | 3 秒语音克隆（含 x-vector only / FT 底座）|
| custom-voice-0.6b | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | 低配预设音色 |
| base-0.6b | Qwen/Qwen3-TTS-12Hz-0.6B-Base | 低配克隆 |

- 语言支持（官方）：中英日韩德法俄葡西意（10 语种 + Auto）
- 合计磁盘占用约 15GB；默认下载目录 `models/`（项目内）或 `QWEN3_TTS_ROCM_MODELS_DIR` 指定
- 下行通道：ModelScope SDK 优先，失败自动回退 HF_ENDPOINT=https://hf-mirror.com

## 4. 架构

```
┌────────────────────────────────────────────────────────┐
│ Demo 层   src/qwen3_tts_rocm/demo/app.py                │
│           qwen3-tts-rocm-demo CLI 入口                  │
├────────────────────────────────────────────────────────┤
│ 适配层    loader.load()│models(注册表+双源下载)          │
│           env(rocm_check)│patch(预留)                   │
├────────────────────────────────────────────────────────┤
│ 官方层    qwen-tts==0.1.1（PyPI 原样，源码零改动）        │
│           全部公开 API 可直接使用                        │
├────────────────────────────────────────────────────────┤
│ 运行时    torch 2.12.0+rocm7.14.0 [device-gfx1151]      │
│           transformers==4.57.3 · gradio · librosa …     │
└────────────────────────────────────────────────────────┘
```

核心原则：**loader.load() 只做参数智能填充，返回官方原生 `Qwen3TTSModel` 实例**。不强包装、不代理方法调用——用户可直接照抄官方文档全部示例。

## 5. 组件规格

### 5.1 env.py — 环境探测与诊断
```python
@dataclass
class EnvReport:
    rocm_wheel_version: str | None      # torch.version.hip，如 "7.14.0"
    gpus: list[GpuInfo]                 # 名称/arch(gfxXXXX)/CU 数/VISIBLE 状态
    driver_ok: bool; warnings: list[str]; errors: list[str]

def rocm_check(verbose=True) -> EnvReport      # 分级诊断：ERROR/WARN/INFO + 中文修复建议
def pick_device(preference="auto") -> str     # "auto"→"cuda:0"(ROCm)/"cpu"，显存<阈值时给 WARN
def require_rocm_torch() -> None              # 未装 HIP torch 时抛带指引的 RuntimeError
CLI: qwen3-tts-rocm-check（ Humans 排障第一入口 ）
```
诊断项：HIP 是否可用、设备可见性（HSA_OVERRIDE/GPU_VISIBLE 相关变量冲突提醒）、/dev/kfd 访问权限组、GTT 大小建议、wheel 版本 ↔ 系统驱动大版本匹配、flash-attn 缺失说明（预期行为非错误）。

### 5.2 models.py — 注册表 + 双源下载
```python
MODELS: dict[str, ModelEntry]         # §3 六仓别名表（repo id、用途、约多少 GB）
def download(alias_or_all, source="modelscope"|"hfmirror"|"auto",
             local_dir=None, resume=True) -> Path
def resolve_path(ref) -> Path          # 官方 repo id｜本地路径｜别名 → 本地缓存路径
```
- auto：先试 ModelScope，超时/失败切 hf-mirror，指数退避重试 2 次
- 断点续传依赖底层 SDK 能力；下载完成写 `.ok` 标记文件防重复下载

### 5.3 loader.py — 加载器（关键公开接口）
```python
def load(model_ref: str|Path,
         device: str = None,            # None→env.pick_device()
         dtype: str|torch.dtype = "bfloat16",
         attn_implementation: str = None) -> Qwen3TTSModel
                                        # None→自动："sdpa"（GPU）/官方默认
```
行为：
1. `resolve_path()` 把别名/repo id 解析成本地路径（未下载则先 download）
2. 若用户显式传 `attn_implementation="flash_attention_2"` 而 flash-atton ROCm 不可用 → 抛错并给出解释与替代值（不静默降级）
3. 其余 kwargs 原样透传官方 `from_pretrained`（保证 device_map/dtype 等所有官方用法不变）
4. 提供 `unload(model)` 辅助函数：del + gc.collect() + empty_cache，供 Demo 切换模型释放内存

### 5.4 patch.py — 运行时兼容补丁点（预留壳）
- 仅当 Spike 后发现官方代码在某处不兼容且可 monkey-patch 时启用
- 每个补丁必须：针对官方精确版本号断言（`qwen_tts.__version__ == "0.1.1"`）+ 单元测试 + 文档记录原因
- 当前预期：空实现（vendor 升级路径保险丝）

### 5.5 demo/app.py — 增强版 Gradio 应用

页签结构（复刻官方 + 增强）：

| 页签 | 对应官方 API | 控件 |
|---|---|---|
| ① 语音克隆 | generate_voice_clone / create_voice_clone_prompt | 参考音频（上传/**麦克风录音**）+参考文本、x-vector only 开关、目标文本、语种下拉；子区 A「克隆并合成」；子区 B「保存/加载音色 .pt」（复刻官方 Save/Load Voice Tab，格式与其 demo.py 相同） |
| ② 预设音色 | generate_custom_voice | 9 音色下拉（来自 get_supported_speakers）、10 语种 Auto、指令控制文本框 |
| ③ 音色设计 | generate_voice_design | 自然语言音色描述、语种、目标文本 |
| ④ 编解码器（增强） | Qwen3TTSTokenizer.encode/decode | 任意外部音频→编码码率信息展示→解码还原试听下载 |
| 全局侧栏 | — | 模型切换器（别名下拉+加载状态，懒加载、驻留一个 TTS 模型+tokenizer，切换先 unload）、高级采样参数折叠面板（max_new_tokens/temp/top_k/top_p/repetition_penalty/subtalker_*）、GPU 显存指示条 |

全局功能：合成历史列表（每条可试听/下载 wav/删除）、示例参考音频资产若干（assets/，短句中英文）、错误信息双语展示、页脚免责声明（沿用官方文案双语版）。

技术要点：
- 构建：`build_demo()` 分解成小函数，事件回调独立模块便于 pytest 直调（不经浏览器即可测业务逻辑）
- 录音需 HTTPS 或 localhost：README 提供 mkcert/openssl 自签两条命令的 HTTPS Notes（沿官方文档改写）
- 启动 CLI：`qwen3-tts-rocm-demo [--models ...] [--ip --port --share/--no-share --ssl-* --concurrency]`，参数面兼容官方 demo.py 同名参数语义

### 5.6 scripts/
| 脚本 | 功能 |
|---|---|
| install.sh | uv 建 .venv → AMD index 装 torch 三件套 → PyPI 装 -e ".[demo]" → verify_gpu 冒烟；--with-models 可选一步到位下载 |
| download_models.sh | 循环六仓 download(auto)，打印体积进度 |
| run_demo.sh | 激活 venv 启动增强 Demo :8000 |
| verify_gpu.sh | 一行冒烟：torch hip 可见 + bf16 matmul + sdpa attention 通过即退出 0 |

## 6. 错误处理策略

| 场景 | 行为 |
|---|---|
| torch 非 HIP 版/CUDA 版误装 | rocm_check ERROR + 重装命令提示 |
| GPU 不可见（kfd 权限等） | ERROR + 提示 groups video/render、HSA 变量检查清单 |
| 网络下载失败 | 双源回退 + 重试；最终失败给出手动下载指引（列出 URL） |
| 显式 flash_attention_2 | 抛错并解释 ROCm 官方无 wheel，推荐不传或 "sdpa" |
| OOM | 提示减 max_new_tokens / 换 0.6B / unload 其它模型；统一内存下 GTT 建议值 |
| Demo 回调异常 | 捕获后 UI 状态栏双语显示 type(e)+msg，不崩进程 |
| 官方版本漂移 | patch/loader 对 `__version__` 断言，未知版本打 WARN 不阻塞（v0.1.x 兼容承诺） |

## 7. 测试策略（TDD）

- **单元层（无 GPU，CI 可跑）**：MODELS 注册表完整性、resolve_path 解析矩阵、EnvReport 构造、gradio build_demo 各页签构建快照式断言、demo 回调的业务逻辑（mock 模型对象注入）
- **GPU 集成层（@pytest.mark.gpu，本地跑）**：
  - test_loader_all_models.py：六仓逐一 load/unload 冒烟断言
  - test_generate.py：三大功能单/批量、指令控制、采样参数透传生效（不同 temperature 输出波形不同）、WaveformSanity（长度>0、非全静音、能量>阈值、sr 匹配）
  - test_voice_clone_workflow.py：clone→create_voice_clone_prompt→重复使用；x-vector only；save(.pt)/load 复用与官方 demo.py 格式互通
  - test_tokenizer_codec.py：encode/decode 往返误差断言 + 五个 getter 值校验
  - test_official_demo_parity.py：**官方原版 `qwen_tts.cli.demo.build_demo` 在本 venv 可构建并可执行一次回调**——保真度证据
- **验收标准**：全部测试绿 + evidence/ 留存：六仓合成样本 wav、RTF 基准数据、rocm_check 报告截图文本
- 性能基准脚本 scripts/benchmark.py：测量 RTF（合成时长/耗时）与首批延迟，结果写入 docs/benchmarks.md

## 8. 发布工程

| 产物 | 说明 |
|---|---|
| GitHub 仓库 | 名 Qwen3-TTS-ROCm；本机网络不通 github.com——本地 git 准备好后由用户推送（README 假定该流程，CONTRIBUTING 写明） |
| README.md / README_CN.md | 英主中文辅：徽章、特性表、30 秒 Quickstart、三行代码最小示例、Demo 截图占位（实施时替换真实截图）、故障排查表、致谢 Qwen 团队 |
| CHANGELOG.md | Keep-a-Changelog 格式，v0.1.0 首发 |
| LICENSE | Apache-2.0（与上游一致），NOTICE 注明衍生自 QwenLM/Qwen3-TTS 及各组件版权归属 |
| Docker | docker/Dockerfile：Ubuntu 24.04 基座 + 与 install.sh 相同的双 index 步骤；运行需 `--device /dev/kfd --device /dev/dri --group-add video/render`；文档注明 gfx1151 统一内存形态 |
| PyPI | 包名 qwen3-tts-rocm（sdist+wheel）。metadata 无法表达私有 index——install 说明里明确"先 AMD index 后 PyPI"两步法；依赖声明 qwen-tts==0.1.1 |
| CI (.github/workflows/ci.yml) | CPU-only：ruff check + pytest -m "not gpu" + 构建发布 dry-run；GPU job 附本地脚本说明（GH runner 无 ROCm GPU） |

## 9. 实施顺序约束（供 writing-plans 使用）

1. **Spike（最先，半天内出结论）**：venv + AMD wheels 装通 → verify_gpu 三验证（HIP matmul/sdpa/bf16）→ 装 qwen-tts → tokenizer 仓冒烟 encode/decode。失败项触发路线升级评估
2. tests 先行（按 §7 结构逐个红→绿）
3. env/models/loader 实现（依赖 Spike 结论）
4. demo/app.py 增强 Demo（可 mock 驱动先行）
5. scripts/docs/README/Docker/CI
6. 收尾：benchmark 采数 → evidence 存档 → 打 tag v0.1.0

## 10. 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| AMD wheel 与 system ROCm 7.2.1 用户态互扰 | 中 | Spike 首验；必要时 venv 内限定 PATH/LD_LIBRARY_PATH 引导至 wheel 自带运行时 |
| transformers 4.57.3 × torch 2.12 不兼容 | 低 | Spike 第 2 步即暴露；pinned 版本组合换 torch 2.11.0+rocm7.14.0 备选 |
| gfx1151 SDPA/bf16 数值问题 | 中 | WaveformSanity 测试兜底；备选 attn=eager；记录进 troubleshooting |
| ModelScope 大文件下载不稳 | 低 | 断点续传 + hf-mirror 回退 + 手动指引 |
| 上游 v0.1.x 小版本变更 | 低 | patch.py 版本断言 + CI 锁版本 |
