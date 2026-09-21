# Reference-closure ground truth — 2026-09-21 (Task 0, program v0.2)

Baseline evidence captured BEFORE any modification of the program. Read-only
audit: no code, tests, docs, or upstream sources were modified; no model
weights were loaded; no test suite was executed (collection only). This file
is the citation target for every later task brief of the Radeon Reference
Closure v0.2 program.

Header facts (all verbatim from the commands cited in the sections below):

| Fact | Value |
|---|---|
| Date | 2026-09-21 |
| Project (AIwork4me/Qwen3-TTS-ROCm, branch main) HEAD | `558d45254579b7349129eeb4d3b3f0b5793a25e2` |
| Upstream pinned local clone (`.upstream/Qwen3-TTS`, branch main) HEAD | `022e286b98fbec7e1e916cb940cdf532cd9f488e` |
| Upstream remote HEAD (`git ls-remote … HEAD`) | `022e286b98fbec7e1e916cb940cdf532cd9f488e` — **identical, no drift** |
| qwen-tts installed (venv) | `0.1.1` |
| Local project version (`pyproject.toml`) | `0.2.0` (unreleased; latest GitHub release `v0.1.0`) |
| Python (venv) | `Python 3.12.3` |
| torch / HIP | `2.12.0+rocm7.14.0` / `7.14.60850` |
| GPU | `AMD Radeon 8060S Graphics`, gfx target `gfx1151` |
| Test collection (CPU `-m 'not gpu'`) | `267/305 tests collected (38 deselected)` |
| Test collection (GPU `-m gpu`) | `38/305 tests collected (267 deselected)` |
| Upstream issue #372 state | `OPEN` (0 comments) |
| `#372` root cause in pinned `sft_12hz.py` | **still present** — line 51, `attn_implementation="flash_attention_2",` |

---

## 1. Project ground truth

Command: `git rev-parse HEAD` (repo root)

```
558d45254579b7349129eeb4d3b3f0b5793a25e2
```

Command: `git status --porcelain` (repo root)

```
(empty)
```

Working tree fully clean before the program's first commit (the plan/spec
commits `953e01e`, `558d452` are already in). Recent history at audit time:

```
558d452 2026-09-21 13:05:50 +0800 docs(plans): radeon reference closure v0.2 implementation plan (20 gated tasks, brief-mapped, user-gated PR + release)
953e01e 2026-09-21 12:58:09 +0800 docs(specs): radeon reference closure v0.2 design (tasks 0-9+12-20; 10-11 deferred; user-gated upstream PR; prepare-only GPU CI)
bd8d613 2026-09-21 09:34:05 +0800 docs: consistency wave from independent docs audit — 12 fixes (...)
```

## 2. Upstream ground truth

Command: `git -C .upstream/Qwen3-TTS rev-parse HEAD`

```
022e286b98fbec7e1e916cb940cdf532cd9f488e
```

`git -C .upstream/Qwen3-TTS status --porcelain` → empty (clone pristine; the
2026-09-20 fine-tune smoke restored it). Branch: `main`. Commit:
`022e286b 2026-03-17 14:38:41 +0800 fix finetuning bug`.

Command: `git ls-remote https://github.com/QwenLM/Qwen3-TTS HEAD`

```
022e286b98fbec7e1e916cb940cdf532cd9f488e	HEAD
```

**Pinned local SHA == remote HEAD — upstream has NOT moved since the
2026-09-20 audit. No fetch/checkout performed (none needed).**

## 3. Versions

Command: `.venv/bin/pip show qwen-tts | head -2`

```
Name: qwen-tts
Version: 0.1.1
```

Command: `grep '^version' pyproject.toml`

```
version = "0.2.0"
```

So: the upstream runtime package is `qwen-tts 0.1.1` (PyPI, unmodified); the
local project declares `0.2.0` in `pyproject.toml` (development version, not
yet released).

Command: `gh release list --limit 3` (current repo, AIwork4me/Qwen3-TTS-ROCm)

```
Qwen3-TTS-ROCm v0.1.0	Latest	v0.1.0	2026-08-27T23:45:32Z
```

Exactly one release exists (`v0.1.0`, marked Latest, published
2026-08-27T23:45:32Z) — the `--limit 3` asked for more but only one exists.

## 4. Upstream issue #372 (verbatim)

Command: `gh issue view 372 --repo QwenLM/Qwen3-TTS --json state,title,comments`

```json
{"comments":[],"state":"OPEN","title":"finetuning/sft_12hz.py hard-codes attn_implementation=\"flash_attention_2\" — official fine-tuning fails out-of-the-box on ROCm (no flash-attn build)"}
```

- `state`: **`OPEN`** (verbatim)
- `title`: `finetuning/sft_12hz.py hard-codes attn_implementation="flash_attention_2" — official fine-tuning fails out-of-the-box on ROCm (no flash-attn build)`
- `comments`: `[]` — zero comments as of 2026-09-21 (filed 2026-09-21, per
  `evidence/upstream-issue-2026-09-21.txt`).

No upstream maintainer response yet; the program's user-gated PR path remains
the actionable route.

## 5. Test collection (NOT executed — `--co` collect-only)

Command: `.venv/bin/python -m pytest -m 'not gpu' --co -q | tail -3`

```
tests/test_validate_languages.py::test_summarize_results_counts_and_per_language_shape

267/305 tests collected (38 deselected) in 4.09s
```

Command: `.venv/bin/python -m pytest -m gpu --co -q | tail -3`

```
tests/test_voice_workflow.py::test_official_schema_preserved

38/305 tests collected (267 deselected) in 4.07s
```

CPU (not-gpu) = **267**, GPU = **38**, total = **305** — matches the README
claim "305 / 305 on validation host — 267 CPU + 38 real-GPU" for the
collection dimension. No suite was run in this audit.

## 6. Capability matrix as currently in README.md (verbatim rows)

Main matrix (README.md, "Capability matrix (Radeon 8060S · `gfx1151`)"),
header summary table included for context:

| Validation | Result |
|---|---|
| Official model repositories | **6 / 6 load-validated** — 5 TTS checkpoints + tokenizer |
| Automated tests | **305 / 305 on validation host** — 267 CPU + 38 real-GPU |
| Patches to upstream `qwen-tts` | **0** — enforced by a dedicated parity test |
| GPU · ROCm | Radeon 8060S (`gfx1151`) · ROCm 7.14.0 (`torch 2.12.0+rocm7.14.0`) |
| Precision / attention | bfloat16 · PyTorch SDPA — FlashAttention not used in the validated stack |
| Evidence | Verbatim transcripts in `evidence/` |

| Capability | Model | Radeon status | Evidence |
|---|---|---|---|
| CustomVoice generation (preset / custom speakers) | 1.7B | ✅ E2E validated | `gen-customvoice.txt` · RTF in `benchmark.json` |
| CustomVoice generation | 0.6B | ✅ E2E validated | `gpu-suite-2026-09-20.txt` · `benchmark-06b-2026-09-20.json` |
| VoiceDesign (text-described voice creation) | 1.7B | ✅ E2E validated | `gen-voicedesign.txt` · RTF in `benchmark.json` |
| Voice Clone — zero-shot cloning from reference audio | 1.7B Base | ✅ E2E validated | `gen-voiceclone.txt` · RTF in `benchmark.json` |
| Base family (zero-shot cloning + fine-tuning base) | 0.6B | ✅ E2E validated | `gpu-suite-2026-09-20.txt` · `benchmark-06b-2026-09-20.json` |
| Reusable clone prompt (`create_voice_clone_prompt` → save → load → reuse) | 1.7B & 0.6B Base | ✅ E2E validated | `gen-voiceclone.txt` · `gpu-suite-2026-09-20.txt` |
| Design → Clone → Reuse (Voice Studio one-click flow) | VoiceDesign 1.7B + Base | ✅ E2E validated | `voice-workflow-2026-09-20.txt` · `voice-workflow-2026-09-20.json` |
| 12Hz tokenizer codec (encode → decode roundtrip) | Tokenizer-12Hz | ✅ E2E validated | `tokenizer-codec.txt` |
| Multilingual matrix — all 10 officially supported languages, end to end | 1.7B CustomVoice + VoiceDesign + Base | ✅ E2E validated | `multilingual-matrix.txt` · `multilingual-matrix.json` |
| Fine-tuning (official `finetuning/` SFT workflow) | 1.7B Base | ✅ scoped — **execution-only smoke** (prep → 12 steps → save → reload → sane synthesis; no quality claims) | `finetune-smoke-2026-09-20.txt` · `finetune-smoke-2026-09-20.json` |
| Instruction control on CustomVoice | 0.6B | 🚫 not exposed upstream (wrapper silently ignores `instruct`) — pinned by tests | Task 0 audit: `ground-truth-2026-09-20.md` |
| vLLM-Omni serving | — | 🟡 partial — offline feasibility proven only: one documented offline example on gfx1151, single configuration, isolated venv (no serving, no perf/quality claims) | `vllm-omni-feasibility-2026-09-21.txt` · roadmap |
| True streaming inference | — | 🚫 not exposed upstream — measured 2026-09-21 on gfx1151: qwen-tts 0.1.1's official Python API delivers audio only at completion (exactly 1 chunk every run; TTFB == total wall) | `streaming-2026-09-21.txt` · roadmap |

(Evidence links relativized to `evidence/` here; in README they are full
repo-relative links. Row text verbatim.)

## 7. `evidence/` inventory (45 files)

`find evidence -type f | wc -l` → `45` (including this index `README.md`;
`spike/` holds 3 of them). Complete sorted listing:

```
benchmark-06b-2026-09-20.json          benchmark-06b-2026-09-20.txt
benchmark-06b-2026-09-21.json          benchmark-06b-2026-09-21.txt
benchmark-2026-09-21.json              benchmark-2026-09-21.txt
benchmark.json                         benchmark-run.txt
ci-2026-09-21-8815238.txt              demo-rest-gen-zh.wav
demo-smoke.txt                         docker-build-final.txt
docker-ci.txt                          docker-gpu-gen-2026-08-30.txt
env-check.txt                          finetune-smoke-2026-09-20.json
finetune-smoke-2026-09-20.txt          first-user-journey-2026-08-29.txt
gen-customvoice.txt                    gen-voiceclone.txt
gen-voicedesign.txt                    gpu-suite-2026-08-29-final.txt
gpu-suite-2026-08-29-firstuser.txt     gpu-suite-2026-08-29.txt
gpu-suite-2026-09-20.txt               ground-truth-2026-09-20.md
install-run.txt                        load-smoke.txt
models-dl.txt                          multilingual-matrix.json
multilingual-matrix.txt                official-parity.txt
README.md                              run-demo-expected-failure.txt
spike/gpu-probe.txt                    spike/spike-report.md
spike/tokenizer-smoke.txt              streaming-2026-09-21.json
streaming-2026-09-21.txt               tokenizer-codec.txt
upstream-issue-2026-09-21.txt          vllm-omni-feasibility-2026-09-21.json
vllm-omni-feasibility-2026-09-21.txt   voice-workflow-2026-09-20.json
voice-workflow-2026-09-20.txt
```

(This file — `reference-closure-ground-truth-2026-09-21.md` — becomes the
46th when committed; it is the only new artifact of Task 0.)

## 8. Upstream `finetuning/sft_12hz.py` — verbatim, with line numbers

Source: `.upstream/Qwen3-TTS/finetuning/sft_12hz.py` at pinned SHA
`022e286b98fbec7e1e916cb940cdf532cd9f488e`.
161 lines (`wc -l` → 161).
sha256: `74d4359bff1ac5eddaca99f0cb5a8b76a55ffbd94af0ab50bffd49aee0e5c473`.

```
   1	# coding=utf-8
  2	# Copyright 2026 The Alibaba Qwen team.
  3	# SPDX-License-Identifier: Apache-2.0
  4	#
  5	# Licensed under the Apache License, Version 2.0 (the "License");
  6	# you may not use this file except in compliance with the License.
  7	# You may obtain a copy of the License at
  8	#
  9	#     http://www.apache.org/licenses/LICENSE-2.0
 10	#
 11	# Unless required by applicable law or agreed to in writing, software
 12	# distributed under the License is distributed on an "AS IS" BASIS,
 13	# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 14	# See the License for the specific language governing permissions and
 15	# limitations under the License.
 16	import argparse
 17	import json
 18	import os
 19	import shutil
 20	
 21	import torch
 22	from accelerate import Accelerator
 23	from dataset import TTSDataset
 24	from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
 25	from safetensors.torch import save_file
 26	from torch.optim import AdamW
 27	from torch.utils.data import DataLoader
 28	from transformers import AutoConfig
 29	
 30	target_speaker_embedding = None
 31	def train():
 32	    global target_speaker_embedding
 33	
 34	    parser = argparse.ArgumentParser()
 35	    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
 36	    parser.add_argument("--output_model_path", type=str, default="output")
 37	    parser.add_argument("--train_jsonl", type=str, required=True)
 38	    parser.add_argument("--batch_size", type=int, default=2)
 39	    parser.add_argument("--lr", type=float, default=2e-5)
 40	    parser.add_argument("--num_epochs", type=int, default=3)
 41	    parser.add_argument("--speaker_name", type=str, default="speaker_test")
 42	    args = parser.parse_args()
 43	
 44	    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")
 45	
 46	    MODEL_PATH = args.init_model_path
 47	
 48	    qwen3tts = Qwen3TTSModel.from_pretrained(
 49	        MODEL_PATH,
 50	        torch_dtype=torch.bfloat16,
 51	        attn_implementation="flash_attention_2",
 52	    )
 53	    config = AutoConfig.from_pretrained(MODEL_PATH)
 54	
 55	    train_data = open(args.train_jsonl).readlines()
 56	    train_data = [json.loads(line) for line in train_data]
 57	    dataset = TTSDataset(train_data, qwen3tts.processor, config)
 58	    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
 59	
 60	    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)
 61	
 62	    model, optimizer, train_dataloader = accelerator.prepare(
 63	        qwen3tts.model, optimizer, train_dataloader
 64	    )
 65	
 66	    num_epochs = args.num_epochs
 67	    model.train()
 68	
 69	    for epoch in range(num_epochs):
 70	        for step, batch in enumerate(train_dataloader):
 71	            with accelerator.accumulate(model):
 72	
 73	                input_ids = batch['input_ids']
 74	                codec_ids = batch['codec_ids']
 75	                ref_mels = batch['ref_mels']
 76	                text_embedding_mask = batch['text_embedding_mask']
 77	                codec_embedding_mask = batch['codec_embedding_mask']
 78	                attention_mask = batch['attention_mask']
 79	                codec_0_labels = batch['codec_0_labels']
 80	                codec_mask = batch['codec_mask']
 81	
 82	                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
 83	                if target_speaker_embedding is None:
 84	                    target_speaker_embedding = speaker_embedding
 85	
 86	                input_text_ids = input_ids[:, :, 0]
 87	                input_codec_ids = input_ids[:, :, 1]
 88	
 89	                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
 90	                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
 91	                input_codec_embedding[:, 6, :] = speaker_embedding
 92	
 93	                input_embeddings = input_text_embedding + input_codec_embedding
 94	
 95	                for i in range(1, 16):
 96	                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
 97	                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
 98	                    input_embeddings = input_embeddings + codec_i_embedding
 99	
100	                outputs = model.talker(
101	                    inputs_embeds=input_embeddings[:, :-1, :],
102	                    attention_mask=attention_mask[:, :-1],
103	                    labels=codec_0_labels[:, 1:],
104	                    output_hidden_states=True
105	                )
106	
107	                hidden_states = outputs.hidden_states[0][-1]
108	                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
109	                talker_codec_ids = codec_ids[codec_mask]
110	
111	                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
112	
113	                loss = outputs.loss + 0.3 * sub_talker_loss
114	
115	                accelerator.backward(loss)
116	
117	                if accelerator.sync_gradients:
118	                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
119	
120	                optimizer.step()
121	                optimizer.zero_grad()
122	
123	            if step % 10 == 0:
124	                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
125	
126	        if accelerator.is_main_process:
127	            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
128	            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)
129	
130	            input_config_file = os.path.join(MODEL_PATH, "config.json")
131	            output_config_file = os.path.join(output_dir, "config.json")
132	            with open(input_config_file, 'r', encoding='utf-8') as f:
133	                config_dict = json.load(f)
134	            config_dict["tts_model_type"] = "custom_voice"
135	            talker_config = config_dict.get("talker_config", {})
136	            talker_config["spk_id"] = {
137	                args.speaker_name: 3000
138	            }
139	            talker_config["spk_is_dialect"] = {
140	                args.speaker_name: False
141	            }
142	            config_dict["talker_config"] = talker_config
143	
144	            with open(output_config_file, 'w', encoding='utf-8') as f:
145	                json.dump(config_dict, f, indent=2, ensure_ascii=False)
146	
147	            unwrapped_model = accelerator.unwrap_model(model)
148	            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
149	
150	            drop_prefix = "speaker_encoder"
151	            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
152	            for k in keys_to_drop:
153	                del state_dict[k]
154	
155	            weight = state_dict['talker.model.codec_embedding.weight']
156	            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
157	            save_path = os.path.join(output_dir, "model.safetensors")
158	            save_file(state_dict, save_path)
159	
160	if __name__ == "__main__":
161	    train()
```

## 9. #372 root-cause verification

Command: `grep -n 'flash_attention_2' .upstream/Qwen3-TTS/finetuning/sft_12hz.py`

```
51:        attn_implementation="flash_attention_2",
```

The hard-coded value is at line 51, inside the `Qwen3TTSModel.from_pretrained`
call (lines 48–52), exactly as reported in issue #372. **The root cause still
exists at the pinned upstream HEAD `022e286b`.**

---

## Anomalies and observations

1. **No upstream drift** — `ls-remote` HEAD equals the pinned local clone
   SHA; no fetch/checkout was needed or performed.
2. **Version spread is intentional**: installed upstream runtime `qwen-tts
   0.1.1` vs local `pyproject.toml` `0.2.0` (unreleased; latest release
   `v0.1.0`). Later version-bump tasks must reconcile these consistently.
3. **Only one GitHub release exists** despite `gh release list --limit 3` —
   nothing anomalous, just fewer than three.
4. **#372 remains unanswered** (`comments: []`), so upstream has neither
   fixed nor rejected the report; the pinned clone still carries the bug.
5. **Collection matches the README claim** (305 total = 267 CPU + 38 GPU);
   suites were NOT executed per the audit's collect-only constraint.
