# Task 14 (Quality benchmark foundation) — Fix Round 1 Re-Verification

Scoped re-review of the fix commit `cfef2e7` ("fix(eval): capture pesq IPR notice
verbatim (verifier round 1)") against the four findings of the FAIL verdict at
`c07055a`. Scope: the findings and the fix diff only.

## 1. Verdicts per finding

### Finding 1 (BLOCKING) — pesq IPR notice "captured verbatim" claimed but absent → **ADDRESSED**

- **Correction block present in the committed transcript.**
  `git show cfef2e7:evidence/quality-benchmark-2026-09-21.txt | sed -n '980,1108p'`
  (rc 0): `=== CORRECTION (fix round 1): pesq IPR notice verbatim capture — Section A
  capture failed (sdist layout), notice follows from the still-on-disk sdist ===`
  through `=== CORRECTION (fix round 1) END 2026-09-21T10:43:57Z ===`.
- **Complete notice.** The block contains DEFINITIONS, NOTICE, OWNERS ARE
  (1. British Telecommunications plc (BT), all rights assigned to Psytechnics
  Limited; 2. Royal KPN NV, all rights assigned to OPTICOM GmbH), RESTRICTIONS
  (the two "cannot" items: alter/duplicate/modify/adapt/translate and
  sell/hire/loan/distribute/dispose/commercial use), PERMITTED USE (items 1–3),
  the PESQ LICENCE AGREEMENT paragraphs (ANY OTHER USE … / OEM LICENSE
  AGREEMENTS …), and the full OPTICOM + Psytechnics contact block.
- **Byte-identical to the source.** I extracted the quoted block from the
  committed file and compared it programmatically to
  `.work-quality/pesq-0.0.4/pesq/pesqmod.c` lines 1–96 transcoded
  Windows-1252→UTF-8: **equal, 96 lines, zero diff**. The transcoding is
  disclosed in the block itself ("copied through iconv -f WINDOWS-1252 -t
  UTF-8 … no other change") — verified true.
- **Provenance recomputed and matching.** `sha256sum` on the still-on-disk
  artifacts: sdist `b724b28f73fb638522982bd68e8c3c0957e2f45210639a460233b17aa7fc890b`
  (38702 bytes), source `e760c30214082ee30d19151f50cd5bd90a82affd4bdfb1a62d10b6119d41b685`
  (53375 bytes, mtime 2022-03-12) — all four facts match the block's provenance
  lines exactly.
- **False claims corrected where correction was due.** `evidence/README.md`
  (line 77) now says the notice "is quoted verbatim in the appended fix-round-1
  correction block, with sdist/source sha256 provenance; Section A's own capture
  step had failed on the sdist layout and the failure is recorded in place."
  The task report's fix-round section acknowledges the Section A/README/report
  claims were false and describes the root cause (glob at top level vs sources
  under `pesq-0.0.4/pesq/`). The historical Section B line ~210 claim and the
  Section A failure lines (153–155) remain byte-untouched, per the program's
  no-hand-edit policy (the same policy `evidence/README.md`'s Errata section
  documents for `finetune-smoke-2026-09-20.txt`), superseded by the appended,
  clearly-marked correction — exactly the remediation the prior verification
  prescribed (correction appended tee -a-style "as was done for smoke-1b and
  venv-import-verify-2, and fix the README claim").
- **Runtime-device statement included** (see Finding 3).

### Finding 2 (Minor) — README "950 lines" vs actual 988 → **ADDRESSED**

`grep -n "1108 lines" evidence/README.md` rc 0 (line 77: "1108 lines incl. the
fix-round-1 correction block"); `wc -l evidence/quality-benchmark-2026-09-21.txt`
→ 1108 (988 original + 120 appended); `grep "950 lines"` on both the working-tree
and committed (`git show cfef2e7:…`) README → rc 1 (figure gone).

### Finding 3 (Minor) — meta.asr.device records policy string, not runtime device → **ADDRESSED**

The correction block contains the "meta clarification" paragraph: the JSON's
`meta.asr.device` records the POLICY string; "The runtime-resolved device for the
Step 14.5 evidence run was: cuda (whisper tiny on the gfx1151 GPU) … the same
resolution path is visible in gate probe SMOKE-4a above ('tiny loaded on cuda',
rc 0). The JSON itself is machine output and is not being edited." Not editing the
machine-produced JSON and stating the resolved value in the committed evidence is
the right fix; the README row also carries the clarification pointer.

### Finding 4 (Minor) — task report "push FAILED" stale → **ADDRESSED**

The report gains a "Correction (fix round 1) — push status" section: the push
was later completed, remote `origin/main` = `c07055a`, `main` no longer ahead.
Verified now: `git rev-parse origin/main` → `cfef2e7` — i.e. the fix commit
itself has since been pushed on top of c07055a, further confirming push health.

## 2. New breakage in the fix diff → **NONE**

- `git diff c07055a..cfef2e7 --numstat` (rc 0):
  `120 0 evidence/quality-benchmark-2026-09-21.txt` (append-only) and
  `1 1 evidence/README.md` (the permitted +/− row rewrite). No other files
  touched; working tree `git diff cfef2e7 -- evidence/` → 0 lines (on-disk ==
  committed).
- **UTF-8 valid, no Windows-1252 residue.** Strict decode of the on-disk file
  (87401 bytes) succeeds (rc 0). A naive scan for bytes 0x91/0x92 found 5
  occurrences — all are the middle byte of 3-byte UTF-8 CJK sequences in the
  verbatim Chinese transcripts (each immediately preceded by a 0xE4/0xE5 lead
  byte; e.g. 中/我/末); a raw Windows-1252 0x91/0x92 would have failed strict
  decoding. The disclosed redone-with-iconv iteration left no residue.
- The README's rewritten row makes only claims I verified: 1108 lines (yes),
  quoted-verbatim-in-correction-block with sha256 provenance (yes, byte-check),
  Section A failure recorded in place (yes, lines 153–155 intact), runtime-device
  statement (yes), driver list +`fix1_ipr_correction.sh` (present on disk in
  gitignored `.work-quality/`).
- The transcript's pre-existing content above the append is untouched (the diff
  hunk begins only after the final RC line of the original file).

## 3. Commands executed and exit codes

1. `git show cfef2e7:evidence/quality-benchmark-2026-09-21.txt | sed -n '980,1108p'` — rc 0; correction block + complete notice + provenance + device statement present. (Also `sed -n '1,50p'` region inspected via the review diff.)
2. `git diff c07055a..cfef2e7 --numstat` — rc 0; `120 0` txt, `1 1` README.
3. `python3 -c "…open('evidence/…txt', encoding='utf-8').read(); strict decode('utf-8')"` — rc 0 (87401 bytes, utf8-ok). First attempt with bare `python` failed rc 127 (not on PATH); retried with `python3` as above. Byte-context scan of all 0x91/0x92 offsets: all CJK continuation bytes.
4. `grep -n "1108 lines" evidence/README.md` — rc 0; `wc -l` → 1108; `grep "950 lines"` (worktree + committed) — rc 1 (absent).
5. `sha256sum .work-quality/pesq-0.0.4.tar.gz .work-quality/pesq-0.0.4/pesq/pesqmod.c` — rc 0; both match the transcript provenance lines; sizes/mtime match.
6. Programmatic byte-compare: quoted block (from `git show cfef2e7:…`) vs pesqmod.c lines 1–96 (cp1252→UTF-8) — identical (True); all required section needles (DEFINITIONS/OWNERS/RESTRICTIONS/PERMITTED USE/LICENCE AGREEMENT/contact) present.
7. `.venv/bin/python -m pytest tests/test_quality_eval.py -q` — **rc 0, 28 passed, 2 warnings, 3.44 s** (matches the fix report; CPU-suite 295/38 recorded in the fix report, quick suite re-run by me as instructed, full-suite count not re-run this round).
8. `git rev-parse origin/main` → cfef2e7; `git log --oneline -3` → HEAD=cfef2e7 on main; `git status --porcelain` → only the (untracked) prior verification report; `ls .work-quality/fix1_ipr_correction.sh` → present.

## 4. Files

- Committed fix (reviewed via diff package `review-c07055a..cfef2e7.diff` and directly):
  `/home/amd/Desktop/Qwen3-TTS-ROCm/evidence/quality-benchmark-2026-09-21.txt` (1108 lines),
  `/home/amd/Desktop/Qwen3-TTS-ROCm/evidence/README.md` (line 77 row rewritten).
- Cross-checked, untracked scratch: `/home/amd/Desktop/Qwen3-TTS-ROCm/.work-quality/pesq-0.0.4/pesq/pesqmod.c`,
  `/home/amd/Desktop/Qwen3-TTS-ROCm/.work-quality/pesq-0.0.4.tar.gz`,
  `/home/amd/Desktop/Qwen3-TTS-ROCm/.work-quality/fix1_ipr_correction.sh`.
- Implementer fix report: `/home/amd/Desktop/Qwen3-TTS-ROCm/.superpowers/sdd/2026-09-21-radeon-reference-closure-v0.2/task-14-report.md` (fix-round sections at end).
- Prior verification: `/home/amd/Desktop/Qwen3-TTS-ROCm/docs/superpowers/reports/rc02-task-14-verification.md`.

## 5. Final verdict

**ALL ADDRESSED — no new breakage in the fix diff.** The blocking evidence gap
is closed in the committed repository by an append-only, provenance-backed,
byte-accurate verbatim capture of the IPR notice; all three minors are corrected
(README figure, runtime-device statement, push status). The covering quick test
suite reproduces (28 passed, rc 0).
