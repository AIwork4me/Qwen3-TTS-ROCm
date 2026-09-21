# Qwen3-TTS issue #372 — root cause and controlled isolation experiment (2026-09-21, Task 2)

Program: Radeon Reference Closure v0.2, Task 2 (brief 2). This document is the
causal-chain record Tasks 3/8/9 cite. Machine-generated verbatim transcript of
the full experiment: `.work-finetune/iso-driver.log` (gitignored scratch;
sha256 `05be3ad742b3b3038ebacb1c77ffcba585d84ca30eb84f29092766aef038bdef`,
311 lines; key excerpts embedded below — nothing hand-edited).

| Fact | Value |
|---|---|
| Date | 2026-09-21 (experiment window 13:39:49–13:46 +08:00) |
| Downstream repo HEAD (branch main) | `5a32ac8` (Task 1 verifier-PASS commit) |
| Upstream pinned SHA (`.upstream/Qwen3-TTS`, main) | `022e286b98fbec7e1e916cb940cdf532cd9f488e` |
| FIX worktree (`.upstream/Qwen3-TTS-fix`) | branch `fix/finetuning-attn-implementation` @ the same pinned SHA `022e286b…` |
| Environment | Python 3.12.3 · torch `2.12.0+rocm7.14.0` · HIP `7.14.60850` · qwen-tts `0.1.1` · transformers `4.57.3` · accelerate `1.12.0` · flash-attn **NOT INSTALLED** · GPU `AMD Radeon 8060S Graphics` (gfx1151) |
| Experiment outcome | **Full chain PASSED, exit 0 at every stage — NO second blocker** |
| Worktree after closeout | restored pristine (`status --porcelain` empty, HEAD `022e286b…`, root-cause line 51 back to `flash_attention_2`) |

---

## 1. Causal chain

Failure leg (control, Task 1, verifier PASS —
`evidence/upstream-372-pristine-failure-run1.txt` / `-run2.txt`):

1. Upstream `finetuning/sft_12hz.py` at pinned SHA `022e286b` calls
   `Qwen3TTSModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
   attn_implementation="flash_attention_2")` — **the hard-coded value is at
   line 51** (verbatim source: Task 0 §8 of
   `evidence/reference-closure-ground-truth-2026-09-21.md`; re-verified in the
   pristine clone and the FIX worktree before the experiment).
2. The installed wrapper forwards `**kwargs` as-is into
   `AutoModel.from_pretrained` (`.venv/.../qwen_tts/inference/qwen3_tts_model.py:83-86`,
   `:112`), so `attn_implementation="flash_attention_2"` reaches transformers'
   model init.
3. transformers `PreTrainedModel.__init__` runs
   `_check_and_adjust_attn_implementation` (transformers `modeling_utils.py:2076`)
   → `get_correct_attn_implementation` (`:2686`, `:2714`) →
   `_flash_attn_2_can_dispatch` (`:2422`), which raises
   `ImportError: FlashAttention2 has been toggled on, but it cannot be used due
   to the following error: the package flash_attn seems to be not installed. …`
   — verbatim in both Task 1 transcripts (run1 line 386, run2 analogous,
   `CLASS-MATCH-VERBATIM` verified). Exit 1, process dies **at model init,
   before any training compute, dataset iteration, or optimizer work**.
4. The validated ROCm wheel stack ships no flash-attn build (probe: `flash-attn
   NOT INSTALLED`; also `src/qwen3_tts_rocm/loader.py:180-184` probes
   `find_spec("flash_attn")`). Therefore every user of the official
   fine-tuning script on this stack hits step 3 unconditionally.

Fix leg (this task, controlled): changing **exactly the one token**
`flash_attention_2` → `sdpa` at that same line 51, in a worktree at the same
pinned SHA, lets the identical invocation chain complete end-to-end:
dataset prep → official `prepare_data.py` → **`python finetuning/sft_12hz.py`
directly (exit 0, 12 optimizer steps, both per-epoch checkpoints saved)** →
reload through the official `Qwen3TTSModel.from_pretrained` API → synthesis →
finite, non-silent, 24 kHz waveform (§4 below). No other change anywhere.

**Root cause statement:** the failure is a *loader-configuration* choice
hard-coded in the fine-tuning script (`sft_12hz.py:51`), not a capability of
the model, not a data/schema problem, and not a second hidden flash-attn
dependency in the workflow. The same model binaries execute and train under
`sdpa` on the same stack.

## 2. The 8 brief questions, answered from source

**Q1 — exact line of the FA2 selection.**
`finetuning/sft_12hz.py:51`: `attn_implementation="flash_attention_2",`
inside the `Qwen3TTSModel.from_pretrained(` call spanning lines 48–52 (grep
output verbatim in Task 0 §9: `51:        attn_implementation="flash_attention_2",`).

**Q2 — configurability elsewhere.**
The fine-tuning script itself exposes **no** flag: its argparse
(`sft_12hz.py:34-42`) offers only `--init_model_path --output_model_path
--train_jsonl --batch_size --lr --num_epochs --speaker_name` — the attention
value is unreachable from the CLI. Elsewhere in upstream the value IS
configurable: the inference wrapper's `from_pretrained(**kwargs)` forwards
anything (`qwen_tts/inference/qwen3_tts_model.py:83-86, :100-101, :112` —
docstring names `attn_implementation` as a typical kwarg); the upstream CLI
demo selects it (`qwen_tts/cli/demo.py:104-108` defines
`--flash-attn/--no-flash-attn`; `demo.py:611-612` maps it to
`attn_impl = "flash_attention_2" if args.flash_attn else None`). Upstream
docs/examples *print* `flash_attention_2` (`README.md:161,214,254,302,319`,
`finetuning/README.md:78`, all four `examples/*.py`), but those are examples,
not enforcement. Only `sft_12hz.py:51` hard-codes it in executable code.

**Q3 — passed directly to `Qwen3TTSModel.from_pretrained`?**
**Yes** — it is a keyword argument of that exact call: `sft_12hz.py:48-52`,
value on line 51 (Q1).

**Q4 — does official inference allow SDPA/alternates?**
**Yes, three independent ways.**
(a) The wrapper accepts any transformers-supported value via kwargs forwarding
(`qwen3_tts_model.py:83-86, :112`); the model core resolves/keeps it at
`qwen_tts/core/models/modeling_qwen3_tts.py:1872-1874, :1888`
(`kwargs.pop("attn_implementation", None)`; falls back to the config's stored
value) and dispatches generically through
`ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]` unless `eager`
(`modeling_qwen3_tts.py:788-789`, `:941-942`) — `sdpa` is a member of that
registry in transformers 4.57.3. The upstream demo's `--no-flash-attn`
(`demo.py:104-108`) passes `attn_implementation=None`, i.e. the official CLI
itself supports running without FA2.
(b) Our loader has SDPA as the HIP default:
`src/qwen3_tts_rocm/loader.py:81` (`_HIP_DEFAULT_ATTN = "sdpa"`) and
`resolve_attn` (`loader.py:201-224`: explicit arg wins, a requested
`flash_attention_2` without an importable flash-attn raises a guided error,
otherwise a HIP GPU gets `"sdpa"`). The whole validated inference stack of this
repo (capability matrix, 305-test suite) runs on that default.
(c) Empirically in this experiment: the official API with the attn argument
**omitted** resolved `config._attn_implementation = 'sdpa'` and produced a
sane waveform (§4 phase D).

**Q5 — does model execution itself require FA2, or is it a loader choice?**
**A loader-configuration choice.** The ImportError is raised inside
transformers' *init-time dispatch check* (traceback: `modeling_utils.py:2076`
→ `:2686` → `:2714` → `:2422`), before any attention kernel would run — i.e.
FA2 fails as a *requested configuration*, not as a *model capability*. The
model code path is attention-implementation-generic
(`modeling_qwen3_tts.py:788-789/941-942`); `_supports_flash_attn = True`
class flags (e.g. installed `modeling_qwen3_tts_tokenizer_v2.py:153, :917`)
are capability declarations, not requirements. Execution evidence: the entire
validated inference stack of this repo runs SDPA on gfx1151, and the
2026-09-20 smoke (`evidence/finetune-smoke-2026-09-20.txt` phases 4.5/4.6)
trained 12 optimizer steps and reloaded under the same one-line sdpa override
(peak 18.02 GiB in-process) — now independently re-proven by this task's
controlled run.

**Q6 — does replacing ONLY the attention implementation allow training to proceed?**
**Yes — demonstrated, not inferred** (§4): with the single-token diff and
nothing else changed, `python finetuning/sft_12hz.py …` exited 0, printed the
script's own `Epoch 0 | Step 0 | Loss: 13.2097` and `Epoch 1 | Step 0 | Loss:
11.9999` lines, and wrote both `checkpoint-epoch-0/` and `checkpoint-epoch-1/`
(each a full copytree + converted `config.json` + `model.safetensors`
3,833,402,520 bytes).

**Q7 — any second hidden FA2 dependency later in the workflow?**
**No.** Grep of the whole `finetuning/` dir: `flash` appears only at
`sft_12hz.py:51` (the root cause) and `finetuning/README.md:78` (prose);
`dataset.py` and `prepare_data.py` contain zero flash/attn references (their
imports are librosa/numpy/torch + `qwen_tts.core.models` config/mel only —
`dataset.py:16-23`). Modules `sft_12hz.py` imports beyond that: the installed
`qwen_tts` package, where every flash reference is non-blocking —
(a) `tokenizer_25hz/vq/whisper_encoder.py:28-37` imports `flash_attn` inside
try/except with a graceful manual-PyTorch fallback (this is the harmless
`Warning: flash-attn is not installed…` banner seen in every phase of every
run, including all passing ones); (b) `tokenizer_12hz`'s
`_supports_flash_attn = True` flags and the `FlashAttentionKwargs` typing
import are declarations/typing, never a hard import; (c) `cli/demo.py`'s flag
(Q2). The 12Hz workflow used here does not route through the 25Hz whisper
encoder's flash path at all. Empirically: the full chain (prep → prepare →
train → save → reload → synthesize) completed with **zero** flash-dependent
failures — the strongest possible answer.

**Q8 — does `sdpa` preserve the expected Qwen3-TTS model path?**
**Yes.** (a) The diff touches one token of one kwarg — no architecture,
data, loss, optimizer, or checkpoint-format change (diff in §4). (b) In-process
probe through the exact modified `from_pretrained` call: resolved
`model.config._attn_implementation = 'sdpa'` AND
`talker_config._attn_implementation = 'sdpa'`; `model_type: qwen3_tts`,
`architectures: ['Qwen3TTSForConditionalGeneration']`, dtype bfloat16 — the
standard Qwen3-TTS path. (c) During training, transformers'
`sdpa_attention.py` ROCm warnings fired (only the SDPA code path emits them),
and the training stdout shows the script's own loss logging. (d) The
checkpoint written by the unmodified save logic reloads through the official
API as a normal `custom_voice` model (`tts_model_type: custom_voice`,
`talker_config.spk_id: {'smoke_speaker': 3000}`,
`spk_is_dialect: {'smoke_speaker': False}`) and synthesizes via
`generate_custom_voice` — the documented §4 reload path of
`finetuning/README.md` (with the same one attn-implementation caveat as
everywhere else: its literal example says `flash_attention_2`).

## 3. The controlled isolation experiment — design

- **Control (failure) leg:** Task 1's verifier-PASS double reproduction — the
  identical invocation against byte-pristine upstream at the same SHA, in fresh
  processes/dirs, fails at `from_pretrained` with the flash-attn ImportError,
  exit 1, reproduced twice (`upstream-372-pristine-failure-run1/-run2.txt`).
- **Treatment (fix) leg:** the same chain, same venv, same models, same
  manifest/seed discipline, same args, same small scale (12 samples, batch 2,
  2 epochs → 12 optimizer steps, exactly the 2026-09-20 smoke scale) — run
  fresh in `.work-finetune/iso/` against the FIX worktree
  `.upstream/Qwen3-TTS-fix` whose ONLY delta is the one token at line 51.
- The training command shape is byte-identical to Task 1's failing Phase C
  (`cd <worktree> && .venv/bin/python finetuning/sft_12hz.py …`) — only the
  worktree path (and the one token inside it) differs.
- Supporting prior (not load-bearing for the isolation claim): the 2026-09-20
  smoke already contained one sdpa-override E2E pass via an in-process
  launcher (`finetune-smoke-2026-09-20.txt` phases 4.5–4.6); this task's run
  is the fresh, controlled, direct-invocation repetition.

## 4. The controlled isolation experiment — commands, outputs, result

Verbatim excerpts from `.work-finetune/iso-driver.log` (full log sha256
`05be3ad7…`; raw per-process streams additionally kept under
`.work-finetune/iso/`: `sft-iso-raw.log`, `probe-raw.log`, `reload-raw.log`).
Every GPU command is wrapped by rocm-smi snapshots (GPU serialization rule);
snapshots show `GPU use (%)` and `VRAM Total Used`. GPU residency of the
training leg is recorded three ways: the rocm-smi watcher caught the GPU busy
at **35%** mid-training (baseline 7–8%); the reload leg recorded
in-process `torch.cuda.get_device_name(0): AMD Radeon 8060S Graphics` and
`torch.cuda.mem_get_info: free=75.84 GiB total=80.00 GiB (model resident on
cuda:0)`; and the prior smoke's in-process `torch.cuda.max_memory_allocated`
for the same workload was 18.02 GiB.

### 4.0 The isolation diff (recorded before the run, from `git -C .upstream/Qwen3-TTS-fix diff`)

```diff
diff --git a/finetuning/sft_12hz.py b/finetuning/sft_12hz.py
index c1f3f46..b5eae64 100644
--- a/finetuning/sft_12hz.py
+++ b/finetuning/sft_12hz.py
@@ -48,7 +48,7 @@ def train():
     qwen3tts = Qwen3TTSModel.from_pretrained(
         MODEL_PATH,
         torch_dtype=torch.bfloat16,
-        attn_implementation="flash_attention_2",
+        attn_implementation="sdpa",
     )
     config = AutoConfig.from_pretrained(MODEL_PATH)
```

`git diff --stat`: `1 file changed, 1 insertion(+), 1 deletion(-)` — exactly
one token. (Blob pair `c1f3f46..b5eae64` is identical to the 2026-09-20
smoke's disclosed deviation — independent reproduction of the same edit.)

### 4.1 Phase A — fresh self-generated dataset (exit 0)

```
$ .venv/bin/python scripts/make_finetune_dataset.py --manifest tests/data/finetune_manifest.json --output-dir .work-finetune/iso --speaker serena
[render] took=7.2s sr=24000 lang=English chars=59        (shared ref clip)
[utt] zh01..zh06 (Chinese), en01..en06 (English) — 12 renders, 2.6–4.5 s each
[done] 12 records -> .work-finetune/iso/train_raw.jsonl
[done] shared ref_audio -> .work-finetune/iso/ref_speaker.wav
$ wc -l .work-finetune/iso/train_raw.jsonl
12
```
Per-utterance durations identical to Task 1's seeded renders (zh 2.08–3.60 s /
en 3.12–3.60 s; `reseed-per-render: 1234` in `dataset_report.txt`).

### 4.2 Phase B — official `prepare_data.py` from the FIX worktree (exit 0)

```
$ (cd .upstream/Qwen3-TTS-fix/finetuning && .venv/bin/python prepare_data.py --device cuda:0 \
    --tokenizer_model_path models/Qwen3-TTS-Tokenizer-12Hz \
    --input_jsonl .work-finetune/iso/train_raw.jsonl --output_jsonl .work-finetune/iso/train_with_codes.jsonl)
********
Warning: flash-attn is not installed. Will only run the manual PyTorch version. Please install flash-attn for faster inference.
********
prepare-exit=0
$ wc -l .work-finetune/iso/train_with_codes.jsonl
12
```
(The banner is `whisper_encoder.py`'s graceful fallback — Q7(a) — not an error;
the SoX "not found" ad likewise, as in every prior run.)

### 4.3 Phase C — isolated training, `sft_12hz.py` invoked DIRECTLY (exit 0)

```
$ (cd .upstream/Qwen3-TTS-fix && .venv/bin/python finetuning/sft_12hz.py \
    --init_model_path models/Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path .work-finetune/iso/output \
    --train_jsonl .work-finetune/iso/train_with_codes.jsonl \
    --num_epochs 2 --speaker_name smoke_speaker)
… transformers/integrations/sdpa_attention.py:96: UserWarning: Mem Efficient attention on Current AMD GPU is still experimental …
… transformers/integrations/sdpa_attention.py:96: UserWarning: Flash Efficient attention on Current AMD GPU is still experimental …
Epoch 0 | Step 0 | Loss: 13.2097
Epoch 1 | Step 0 | Loss: 11.9999
sft-iso-exit=0   (exit of the python process itself, NOT of the watcher/tee)
```
- The two `sdpa_attention.py` warnings are emitted only when the SDPA kernels
  actually execute — runtime proof the SDPA path ran during training (Q8c).
- Command shape is identical to Task 1 Phase C (which exits 1 at the same
  point in the pristine clone); the ONLY difference is the worktree's one
  token. **The causal pair is closed.**
- 12 optimizer steps: 12 samples / batch_size 2 = 6 microbatch-steps per epoch
  × 2 epochs (`optimizer.step()` runs per microbatch — upstream lines 120-121
  are not gated on `sync_gradients`, per the 2026-09-20 audit).
- Watcher residency samples: `[watcher 1] GPU use 8%` → `[watcher 2] GPU use
  35%, VRAM 1,415,602,176 B` (mid-run). On this Strix Halo iGPU rocm-smi's
  VRAM counter barely moves (unified memory; torch sees 80 GiB — §4.5), so
  the busy-percentage spike plus in-process stats are the residency record.
- Both checkpoints written by the script's own save logic
  (`checkpoint-epoch-0` 13:41:53, `checkpoint-epoch-1` 13:42:00; each
  `model.safetensors` 3,833,402,520 bytes — full copytree + converted config).

### 4.4 Phase C2 — attn-resolution probe through the EXACT modified call (exit 0)

```
[probe] resolved model.config._attn_implementation = 'sdpa'
[probe] talker_config._attn_implementation = 'sdpa'
[probe] torch.cuda.get_device_name(0): AMD Radeon 8060S Graphics
[probe] torch.cuda.max_memory_allocated: 0.000 GiB   (model on cpu — see note)
[probe] model dtype: torch.bfloat16 device: cpu
[probe] model_type: qwen3_tts | architectures: ['Qwen3TTSForConditionalGeneration']
PROBE-OK
```
Note: the official script's `from_pretrained` (line 48-52) passes no
`device_map`, so the model initially lands on CPU and `accelerator.prepare`
moves it to the GPU (that is upstream behavior, unchanged by the one token) —
hence 0 GiB in this CPU-resident probe; GPU residency of the training leg is
carried by the watcher + phase-D in-process stats + the prior smoke's 18.02
GiB in-process figure for the same workload.

### 4.5 Phase D — official-API reload + synthesis + waveform sanity (exit 0)

```
$ .venv/bin/python .work-finetune/iso_reload_check.py
[iso-reload] official path: Qwen3TTSModel.from_pretrained(device_map='cuda:0', dtype=torch.bfloat16, attn_implementation OMITTED)
[iso-reload] from_pretrained took=2.6s
[iso-reload] resolved model.config._attn_implementation = 'sdpa'
[iso-reload] talker config._attn_implementation = 'sdpa'
[iso-reload] supported speakers: ['smoke_speaker']
[iso-reload] torch.cuda.get_device_name(0): AMD Radeon 8060S Graphics
[iso-reload] torch.cuda.mem_get_info: free=75.84 GiB total=80.00 GiB (model resident on cuda:0, dtype=torch.bfloat16)
[iso-reload] generate_custom_voice took=6.7s sr=24000 n_wavs=1
[iso-reload] WAV-SANE=True finite=True non-silent=True sr=24000 (expect 24000) duration_s=4.00 (bounds 0.5-60) peak=0.2275 rms=0.03978
[iso-reload] wrote .work-finetune/iso/reload_synth_en01.wav
ISO-RELOAD-AND-SYNTHESIS-OK
$ checkpoint config sanity (written by the training script itself)
tts_model_type: custom_voice
talker_config.spk_id: {'smoke_speaker': 3000}
talker_config.spk_is_dialect: {'smoke_speaker': False}
```
Synthesis text was manifest sentence en01 (the same text as the training
clip), speaker `smoke_speaker`, `max_new_tokens=512`, language `English`.
This reload goes through the OFFICIAL wrapper directly (no
`qwen3_tts_rocm.loader`), with the attn argument omitted so the official
default resolution applies — resolving to `sdpa` (Q4c/Q8b).

### 4.6 Result

| Stage | Command | Exit |
|---|---|---|
| A dataset | `scripts/make_finetune_dataset.py --manifest tests/data/finetune_manifest.json --output-dir .work-finetune/iso --speaker serena` | 0 |
| B prepare | `prepare_data.py --device cuda:0 --tokenizer_model_path models/Qwen3-TTS-Tokenizer-12Hz --input_jsonl …train_raw.jsonl --output_jsonl …train_with_codes.jsonl` | 0 |
| C train | `sft_12hz.py --init_model_path models/Qwen3-TTS-12Hz-1.7B-Base --output_model_path .work-finetune/iso/output --train_jsonl …train_with_codes.jsonl --num_epochs 2 --speaker_name smoke_speaker` | **0** (Task 1 control: **1**) |
| C2 probe | modified `from_pretrained` call verbatim | 0 |
| D reload+synth | `iso_reload_check.py` (official `Qwen3TTSModel.from_pretrained`) | 0 |

**Full chain completed. NO second blocker appeared at any stage.** The one
flash-attn `ImportError` seen in the control legs is the only failure on the
path, and the one-token change removes it without introducing any new failure.

## 5. Alternative hypotheses considered and rejected

1. **"The model itself needs FA2" (architecture coupling).** Rejected: the
   ImportError is raised by transformers' init-time *dispatch check*
   (`modeling_utils.py:2422`) before any kernel runs; the model code dispatches
   generically via `ALL_ATTENTION_FUNCTIONS` (`modeling_qwen3_tts.py:788-789`);
   and the full train+synthesize chain ran under `sdpa` (this task) — a model
   that *required* FA2 could not have trained, saved, reloaded, and synthesized.
2. **"A second flash dependency lurks later in the workflow."** Rejected by
   source grep (Q7: `dataset.py`/`prepare_data.py` clean; installed package's
   only flash import is the graceful 25Hz whisper fallback, unused by the 12Hz
   path) and empirically: the chain completed past every later stage
   (data iteration, optimizer, backward, checkpoint save, reload, synthesis).
3. **"The failure is data/schema-related."** Rejected: in both Task 1 control
   runs the failure happens at `from_pretrained` — BEFORE the training JSONL is
   even opened (`sft_12hz.py:55` runs after line 48-52); and the identical
   dataset pipeline (phases A/B) exits 0 in this task too.
4. **"Version skew (qwen-tts 0.1.1 vs upstream repo, or transformers 4.57.3
   behavior change)."** Rejected as the *root* cause: the same installed stack
   is held fixed across the failing control legs and the passing treatment
   leg; the only varying factor is the one token. (Version skew remains
   relevant only as context for why `flash_attention_2` became a hard error —
   transformers' `_flash_attn_2_can_dispatch` import check — not for whether
   the script is at fault for hard-coding it.)
5. **"Installing flash-attn would be the right fix."** Not tested here (out of
   scope: no ROCm flash-attn wheel exists for this validated stack, per the
   project's zero-flash-attn posture), and unnecessary for the causal claim:
   the experiment shows `sdpa` suffices. Whether upstream prefers a default
   switch, a CLI flag, or auto-fallback is a design choice for the upstream PR
   (Task 8) — the evidence here supports "selectable/default-safe", not
   "flash-attn required".

## 6. Remaining uncertainties

- **Execution-scope only.** As with the 2026-09-20 smoke: no claims about
  convergence quality, speaker similarity, loss semantics, or long-run
  stability. Loss values (13.2097 → 11.9999) are quoted verbatim from the
  script's own logging (shuffle order differs run-to-run, so they are not
  comparable to the smoke's 13.6470/15.0211 — and are not interpreted).
- **Single scale/model.** 1.7B Base, 12 samples, 12 optimizer steps, bf16.
  Other checkpoints (0.6B), longer runs, or multi-speaker training were not
  exercised in the isolation experiment.
- **Not tested: flash-attn actually installed.** The claim proven is "sdpa
  removes the blocker", not "flash_attention_2 would also work if a build
  existed".
- **rocm-smi VRAM counter on this APU** barely reflects allocations (unified
  memory; torch reports an 80 GiB pool). Residency is therefore evidenced by
  GPU-busy percentage, in-process torch stats, and file outputs rather than
  the rocm-smi VRAM delta.
- The watcher loop captured 2 of the ~66 s training window (t≈0 and t≈17 s,
  the latter showing 35% GPU busy); it under-sampled the tail (loop exited
  early while the process ran to 13:42:01 — the authoritative exit code came
  from `wait`, and both checkpoint mtimes bound the window). Recorded as-is;
  no claim of continuous residency monitoring.

## 7. Worktree closeout (Task 3 re-applies the change as a commit)

```
$ git -C .upstream/Qwen3-TTS-fix checkout -- .
checkout-exit=0
$ git -C .upstream/Qwen3-TTS-fix status --porcelain   (expect EMPTY)
[status lines: 0]
$ git -C .upstream/Qwen3-TTS-fix rev-parse HEAD
022e286b98fbec7e1e916cb940cdf532cd9f488e
$ grep -n flash_attention_2 .upstream/Qwen3-TTS-fix/finetuning/sft_12hz.py
51:        attn_implementation="flash_attention_2",        (pristine again)
$ git -C .upstream/Qwen3-TTS status --porcelain | wc -l   (pristine clone untouched)
0
$ git -C <repo root> status --porcelain | wc -l           (main repo; .upstream/ + .work-finetune/ gitignored)
0
```

The branch `fix/finetuning-attn-implementation` remains checked out in the
worktree at the pinned SHA, clean, ready for Task 3's commit.

### Errata — recording-level notes on the experiment transcript (log left unedited)

1. Phase A's `generator-exit=0` label reflects the display pipeline's status,
   not a captured python exit code; phase-A success is established by the
   generator's own `[done] 12 records`/`[done] shared ref_audio` lines,
   `wc -l` → 12, and `dataset_report.txt` (12 rows). Phases B/C/C2/D captured
   true process exits (`PIPESTATUS[0]` / `wait` / `$?` on a plain
   non-pipelined call).
2. The Phase A raw-stream tee target (`iso/make-dataset-raw.log`) was opened
   before the generator created the `iso/` directory, so that side-file does
   not exist; the filtered (noise-removed) stream in `iso-driver.log` contains
   every non-MIOpen-noise line of phase A. Phases C/C2/D raw streams were
   captured to files that existed and are kept under `.work-finetune/iso/`.
