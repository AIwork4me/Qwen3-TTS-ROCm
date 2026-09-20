# Ground-truth audit — 2026-09-20

Header facts (all verbatim from the commands cited in the sections below):

| Fact | Value |
|---|---|
| Date | 2026-09-20 |
| Project (AIwork4me/Qwen3-TTS-ROCm, branch main) HEAD | `65dc7b1794c785b004d8929d64c0b2dd17d93438` |
| Upstream (QwenLM/Qwen3-TTS) HEAD SHA | `022e286b98fbec7e1e916cb940cdf532cd9f488e` |
| qwen-tts installed | `0.1.1` |
| Python (venv) | `Python 3.12.3` |
| torch / HIP | `2.12.0+rocm7.14.0` / `7.14.60850` |
| GPU | `AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S`, gfx target `gfx1151` |

Read-only audit. No code, tests, or upstream sources were modified; no model weights were loaded.

---

## 1. Project ground truth

Command: `git -C /home/amd/Desktop/Qwen3-TTS-ROCm rev-parse HEAD`

```
65dc7b1794c785b004d8929d64c0b2dd17d93438
```

Command: `git -C /home/amd/Desktop/Qwen3-TTS-ROCm status --porcelain`

```
 M .gitignore
```

Only entry is ` M .gitignore` — the expected exception per the task brief. `git diff .gitignore` shows it is a single appended line `.superpowers/` (scratch-dir ignore); `.superpowers/` itself does not appear because of that ignore. Working tree is otherwise clean.

## 2. Upstream ground truth

Command: `git ls-remote https://github.com/QwenLM/Qwen3-TTS HEAD`

```
022e286b98fbec7e1e916cb940cdf532cd9f488e	HEAD
```

All upstream file contents in section 4 were fetched at this SHA via
`https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/022e286b98fbec7e1e916cb940cdf532cd9f488e/<path>`.

## 3. qwen-tts version actually installed

Command: `/home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/pip show qwen-tts`

```
Name: qwen-tts
Version: 0.1.1
```

(Full output also reports `Location: /home/amd/Desktop/Qwen3-TTS-ROCm/.venv/lib/python3.12/site-packages`, `Requires: accelerate, einops, gradio, librosa, onnxruntime, soundfile, sox, torchaudio, transformers`, and `Required-by: qwen3-tts-rocm`. The local editable `qwen3_tts_rocm-0.1.0` dist is also installed.)

Pinned deps from `qwen_tts-0.1.1.dist-info/METADATA` (`grep -E "^(Name|Version|Requires-Dist)" .../METADATA`):

```
Name: qwen-tts
Version: 0.1.1
Requires-Dist: transformers==4.57.3
Requires-Dist: accelerate==1.12.0
Requires-Dist: gradio
Requires-Dist: librosa
Requires-Dist: torchaudio
Requires-Dist: soundfile
Requires-Dist: sox
Requires-Dist: onnxruntime
Requires-Dist: einops
```

Console script (`qwen_tts-0.1.1.dist-info/entry_points.txt`):

```
[console_scripts]
qwen-tts-demo = qwen_tts.cli.demo:main
```

## 4. Upstream `finetuning/` layout (at SHA `022e286b98fbec7e1e916cb940cdf532cd9f488e`)

Command: `curl -s https://api.github.com/repos/QwenLM/Qwen3-TTS/contents/finetuning`

Exactly four files, no subdirectories:

| File | Size (bytes) |
|---|---|
| `finetuning/README.md` | 3165 |
| `finetuning/dataset.py` | 8664 |
| `finetuning/prepare_data.py` | 2410 |
| `finetuning/sft_12hz.py` | 6885 |

### README (fetched verbatim from `.../022e286.../finetuning/README.md`)

- Title: "Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base". Scope: "currently supports single-speaker fine-tuning"; "Multi-speaker fine-tuning and other advanced fine-tuning features will be supported in future releases."
- Declared prerequisite: "Please run `pip install qwen-tts` first" — this is the only declared dependency statement; there is **no** `requirements.txt` or similar file in `finetuning/`. (De-facto imports of the scripts: `torch`, `accelerate`, `safetensors`, `transformers`, `librosa`, `numpy`, `qwen_tts`.)
- Input JSONL format: one JSON object per line with keys `audio` (wav path), `text` (transcript), `ref_audio` (reference speaker wav). README: "Strongly recommended: use the same `ref_audio` for all samples."
- Documented data-preparation entry point (README section "2) Prepare data (extract `audio_codes`)"):

  ```
  python prepare_data.py \
    --device cuda:0 \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --input_jsonl train_raw.jsonl \
    --output_jsonl train_with_codes.jsonl
  ```

- Documented training entry point (README section "3) Fine-tune"):

  ```
  python sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path output \
    --train_jsonl train_with_codes.jsonl \
    --batch_size 32 \
    --lr 2e-6 \
    --num_epochs 10 \
    --speaker_name speaker_test
  ```

- Checkpoints written to `output/checkpoint-epoch-0`, `output/checkpoint-epoch-1`, ... (one per epoch).
- Checkpoint reload for inference (README section "4) Quick inference test", verbatim):

  ```python
  import torch
  import soundfile as sf
  from qwen_tts import Qwen3TTSModel

  device = "cuda:0"
  tts = Qwen3TTSModel.from_pretrained(
      "output/checkpoint-epoch-2",
      device_map=device,
      dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
  )

  wavs, sr = tts.generate_custom_voice(
      text="She said she would be here by noon.",
      speaker="speaker_test",
  )
  sf.write("output.wav", wavs[0], sr)
  ```

  I.e. the checkpoint directory is reloaded directly via the standard `Qwen3TTSModel.from_pretrained()` — no separate conversion step.
- README also contains a "One-click shell script example" (bash) chaining `prepare_data.py` then `sft_12hz.py` with `BATCH_SIZE=2`, `LR=2e-5`, `EPOCHS=3`, `SPEAKER_NAME="speaker_1"`.

### `prepare_data.py` — data-preparation script (fetched at the SHA)

argparse CLI, verbatim:

```python
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
parser.add_argument("--input_jsonl", type=str, required=True)
parser.add_argument("--output_jsonl", type=str, required=True)
```

Behavior: loads `Qwen3TTSTokenizer.from_pretrained(args.tokenizer_model_path, device_map=args.device)`, reads `--input_jsonl`, batches `line['audio']` in groups of `BATCH_INFER_NUM = 32` (module-level constant), calls `tokenizer_12hz.encode(batch_audios)`, writes each input line back with a new key `audio_codes` (from `enc_res.audio_codes`, `.cpu().tolist()`) to `--output_jsonl`.

### `sft_12hz.py` — training script (fetched at the SHA)

argparse CLI, verbatim:

```python
parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
parser.add_argument("--output_model_path", type=str, default="output")
parser.add_argument("--train_jsonl", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--speaker_name", type=str, default="speaker_test")
```

Key mechanics (read from the full file):
- `Accelerator` is hard-coded: `accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")` — not CLI-configurable.
- Loads `Qwen3TTSModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")` — `flash_attention_2` is hard-coded.
- Dataset: `TTSDataset(train_data, qwen3tts.processor, config)` from local `dataset.py`; `DataLoader(..., batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)`.
- Optimizer: `AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)`; trains `qwen3tts.model` (talker + speaker encoder) with `loss = outputs.loss + 0.3 * sub_talker_loss`; grad clip 1.0 via `accelerator.clip_grad_norm_`.
- Speaker handling: `speaker_embedding = model.speaker_encoder(ref_mels...).detach()`; the **first batch's** embedding is captured in the module-global `target_speaker_embedding`.
- Checkpoint writing (per epoch, main process only): `shutil.copytree(MODEL_PATH, output/checkpoint-epoch-{epoch}, dirs_exist_ok=True)`; then rewrites `config.json` with `config_dict["tts_model_type"] = "custom_voice"` and `talker_config["spk_id"] = {args.speaker_name: 3000}` and `talker_config["spk_is_dialect"] = {args.speaker_name: False}`; drops all `speaker_encoder*` keys from the state dict; overwrites `state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0]...`; saves the rest as `model.safetensors` (`safetensors.torch.save_file`). So a finetuned Base checkpoint is converted into a CustomVoice-format model keyed at speaker id 3000, reloadable via plain `from_pretrained` (see README snippet above).

### `dataset.py` (fetched at the SHA)

Not a CLI script: defines `TTSDataset(Dataset)` (`__init__(self, data_list, processor, config: Qwen3TTSConfig, lag_num=-1)`) with audio normalization helpers (`librosa.load(x, sr=None, mono=True)`) and the collate producing `input_ids`, `codec_ids`, `ref_mels`, `text_embedding_mask`, `codec_embedding_mask`, `attention_mask`, `codec_0_labels`, `codec_mask` (batch keys as consumed by `sft_12hz.py`). Imports `librosa`, `numpy`, `torch`, `qwen_tts.core.models.{configuration_qwen3_tts, modeling_qwen3_tts}`.

### Requirements/deps declared

No `requirements.txt`/`pyproject`/environment file inside `finetuning/`. Declared: "Please run `pip install qwen-tts` first" (README). Implicit imports across the three .py files: `torch`, `accelerate`, `safetensors`, `transformers`, `librosa`, `numpy`, `qwen_tts`.

## 5. Installed-package runtime facts (`.venv/lib/python3.12/site-packages/qwen_tts/`, source-reading only — no weights loaded)

### 5a. What `get_supported_languages()` returns

It is **not** a static list/constant. Model level — `qwen_tts/core/models/modeling_qwen3_tts.py:1830-1834` (init of `Qwen3TTSForConditionalGeneration`):

```python
self.supported_speakers = self.config.talker_config.spk_id.keys()
self.supported_languages = ["auto"]
for language_id in self.config.talker_config.codec_language_id.keys():
    if "dialect" not in language_id:
        self.supported_languages.append(language_id)
```

`get_supported_languages()` (`modeling_qwen3_tts.py:1852-1853`) returns `self.supported_languages`.

Wrapper level — `qwen_tts/inference/qwen3_tts_model.py:861-877` returns `sorted(set(lowercased))` of the above (`_supported_languages_set`, lines 123-130).

Computed by applying that exact init logic to the six locally downloaded `models/*/config.json` files (JSON read only; no weights loaded):

- All five TTS models (`Qwen3-TTS-12Hz-0.6B-Base`, `0.6B-CustomVoice`, `1.7B-Base`, `1.7B-CustomVoice`, `1.7B-VoiceDesign`) have identical `codec_language_id` keys, so:
  - `model.get_supported_languages()` (insertion order) =

    ```
    ['auto', 'chinese', 'english', 'german', 'italian', 'portuguese', 'spanish', 'japanese', 'korean', 'french', 'russian']
    ```

  - `Qwen3TTSModel.get_supported_languages()` (wrapper, sorted) =

    ```
    ['auto', 'chinese', 'english', 'french', 'german', 'italian', 'japanese', 'korean', 'portuguese', 'russian', 'spanish']
    ```

  (11 entries; dialect-prefixed keys are excluded by the `if "dialect" not in language_id` filter.)

### 5b. `instruct` handling on the 0.6B CustomVoice path

Decisive evidence — `qwen_tts/inference/qwen3_tts_model.py:799-800`, inside `generate_custom_voice()`:

```python
if self.model.tts_model_size in "0b6": # for 0b6 model, instruct is not supported
    instruct = None
```

Verdict: **silently ignored** — for `tts_model_size == "0b6"` the parameter is overwritten with `None` before use. It does **not** raise and does **not** warn (no `warnings.warn`/logger call anywhere on this path; `grep -rn instruct --include="*.py"` over the package finds no other guard). The docstring (lines 751-752: "Optional instruction(s). If None, treated as empty (no instruction).") does not mention the 0.6B drop. Downstream, `None`/`""` instructs become `instruct_ids.append(None)` (lines 820-825) and the model skips instruct embedding (`core/models/modeling_qwen3_tts.py:2076-2080` only embeds non-None instruct ids). Note the guard is a substring test (`in "0b6"`), matching size strings like `"0b6"` but not `"1b7"`.

### 5c. Public API entry points

`qwen_tts/__init__.py` (verbatim, lines 21-24):

```python
from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

__all__ = ["__version__"]
```

- Importable names verified live (`import qwen_tts`): `Qwen3TTSModel`, `VoiceClonePromptItem`, `Qwen3TTSTokenizer` (all present; import succeeds, printing only "Warning: flash-attn is not installed...").
- Oddity recorded verbatim: `__all__ = ["__version__"]` but no `__version__` is assigned anywhere in the package — `getattr(qwen_tts, '__version__', ...)` finds no attribute (verified live), so `from qwen_tts import *` would raise `AttributeError`.
- `Qwen3TTSModel` public methods (`grep -n "    def [a-zA-Z]" qwen3_tts_model.py | grep -v "def _"`): `from_pretrained` (line 83), `create_voice_clone_prompt` (356), `generate_voice_clone` (470), `generate_voice_design` (637), `generate_custom_voice` (732), `get_supported_speakers` (842), `get_supported_languages` (861).
- `VoiceClonePromptItem` (`qwen3_tts_model.py:40-51`): dataclass with fields `ref_code`, `ref_spk_embedding`, `x_vector_only_mode`, `icl_mode`, `ref_text=None`.
- CLI: console script `qwen-tts-demo` → `qwen_tts.cli.demo:main`; `python -m qwen_tts` (`__main__.py`) only prints a pointer to `qwen-tts-demo`.

## 6. Existing validated hardware evidence inventory (`ls -la evidence/`)

Per `evidence/README.md` (self-description: produced on the validation host "AMD Ryzen AI Max+ PRO 395 / Radeon 8060S, gfx1151, ROCm 7.14.0, kernel 6.17.0-1032-oem, Python 3.12", mostly 2026-08-27), plus the file listing. "Real-GPU transcript" = document of commands actually executed on the GPU host:

| File | Capability it proves | Real-GPU transcript? |
|---|---|---|
| `gen-customvoice.txt` | CustomVoice generation path (on-GPU generation suite) | Yes (GPU test run) |
| `gen-voicedesign.txt` | VoiceDesign generation path | Yes |
| `gen-voiceclone.txt` | VoiceClone generation path | Yes |
| `benchmark.json` / `benchmark-run.txt` | RTF benchmarks (per-cell RTF lists; raw session stdout) | Yes (GPU inference measured) |
| `gpu-suite-2026-08-29.txt` / `-final.txt` / `-firstuser.txt` | Full on-GPU pytest suites (25 GPU tests; final also 238/238 CPU+GPU) | Yes |
| `load-smoke.txt` | All-six-model GPU load/unload smoke | Yes |
| `tokenizer-codec.txt` | Official 12Hz tokenizer encode→decode roundtrip on GPU | Yes |
| `official-parity.txt` | Upstream Gradio demo parity incl. one real synthesis | Yes |
| `demo-smoke.txt` + `demo-rest-gen-zh.wav` | Live five-tab demo: HTTP boot, one real REST synthesis (5.52 s zh clip) | Yes |
| `docker-gpu-gen-2026-08-30.txt` | Real in-container GPU synthesis (custom-voice REST, 6.24 s WAV; HIP available, gfx1151) | Yes |
| `docker-build-final.txt` / `docker-ci.txt` | Docker image build + validation ladder | Build/CI records, not synthesis transcripts |
| `models-dl.txt` | Six-repo model download provenance | No (network/download log) |
| `install-run.txt` | `scripts/install.sh` pinned wheel stack install | No (install record) |
| `env-check.txt` | Environment self-check (`qwen3-tts-rocm-check`) | Partial (host env probe) |
| `first-user-journey-2026-08-29.txt` | First-user acceptance journey (hand-compiled summary; scratch transcripts listed in its PROVENANCE) | Summary, marked hand-compiled |
| `run-demo-expected-failure.txt` | The original parity-hypothesis failure record | Manual transcript |
| `tokenizer-codec.txt` (above), `spike/spike-report.md`, `spike/gpu-probe.txt` (`SPIKE-GPU-OK`), `spike/tokenizer-smoke.txt` (`TOKENIZER-OK`) | Day-one feasibility spike (wheel stack, GPU sanity) | gpu-probe: yes |

Not present in `evidence/`: any finetuning/SFT transcript, any 0.6B-specific generation transcript (the generation suites above are named by path, not by model size — model size per suite is not stated in the file listing; `evidence/README.md` does not record one), no multi-speaker finetuning record.

## 7. Environment facts

- Command: `/home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/python --version` →

  ```
  Python 3.12.3
  ```

- Command: `/home/amd/Desktop/Qwen3-TTS-ROCm/.venv/bin/python -c "import torch; print(torch.__version__, torch.version.hip)"` →

  ```
  2.12.0+rocm7.14.0 7.14.60850
  ```

- Command: `rocminfo | grep -E "Marketing Name|gfx" | head` →

  ```
    Marketing Name:          AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S
    Name:                    gfx1151
    Marketing Name:          AMD Radeon Graphics
        Name:                    amdgcn-amd-amdhsa--gfx1151
        Name:                    amdgcn-amd-amdhsa--gfx11-generic
    Marketing Name:          RyzenAI-npu5
  ```

  GPU: AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S; gfx target `gfx1151`.
