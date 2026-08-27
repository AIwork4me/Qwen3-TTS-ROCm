# tests/test_official_demo_parity.py
"""Task 17: official-parity proof -- UNMODIFIED upstream demo on our ROCm stack.

This is the project's flagship guarantee: prove that the stock, unmodified
``qwen_tts.cli.demo.build_demo`` (installed verbatim from upstream into the
venv; see ``.venv/lib/python3.12/site-packages/qwen_tts/cli/demo.py``) both
constructs and EXECUTES against a model object produced by
:func:`qwen3_tts_rocm.loader.load`.  The historical red state for this exact
hypothesis is documented in ``evidence/run-demo-expected-failure.txt``
(stock entry point exploded before Task 5 wired the ROCm shim); today's run
is the green parity artifact the README (Task 20) cites via
``evidence/official-parity.txt``.

Closure-reachability -- which mechanism was chosen and why (binding record)
--------------------------------------------------------------------------
Upstream defines its handlers (``run_instruct`` / ``run_voice_design`` /
``run_voice_clone`` / ``save_prompt`` / ``load_prompt_and_gen``) as CLOSURES
inside :func:`build_demo`; they are NOT attributes of the returned Blocks.
The plan sketched three routes and ranked them.  Measured on gradio 6.17.3:

* ``demo.fns`` is ``dict[int, gradio.block_function.BlockFunction]`` -- each
  entry carries ``.fn`` (the genuine closure object), ``.inputs``, ``.outputs``
  and ``.targets`` -- i.e. Gradio's OWN event registry holds exactly the
  callables that click dispatch would invoke.  NO archaeology needed: this is
  the strongest route (a real closure execution "through Gradio's machinery"),
  so it was taken and there was no need for the module-level-helper fallback.

Test B therefore invokes the registered ``run_instruct`` dependency's ``.fn``
directly with display-style values (the same shape Gradio hands a handler
after component preprocessing); Test C does one REAL synthesis through the
same stored callable and sanity-gates the waveform + prints RTF.

Known-plan deviation (justified): the brief/plan floor asked for
``len(demo.blocks) > 20`` as minimum construction evidence.  Empirically the
genuine upstream ``custom_voice`` layout registers exactly 17 components and
exactly 1 BlockFunction entry on gradio 6.17.3 (the >20 figure only holds for
the base-model two-tab voice-clone layout).  Since closures ARE reachable,
spec §7 row 4's stronger branch ("可执行一次回调" -- one callback executable)
is satisfied instead of the numeric minimum-evidence branch, so the floor is
right-sized to the real graph (``test_upstream_blocks_constructed`` pins
>= 15 plus upstream-specific layout fingerprints: title markdown, checkpoint
line carrying OUR local weights path, disclaimer text, button label).
"""

from __future__ import annotations

import time

import gradio as gr
import numpy as np
import pytest

# The INSTALLED UPSTREAM module, imported straight from site-packages.
import qwen_tts.cli.demo as upstream_demo

from qwen3_tts_rocm import loader, models, testing

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]

#: Short target text so the official closure's token stream stays inside the
#: suite-wide latency guardrail (deliberate + official kwarg via CLI-flag
#: plumbing, NOT an assertion change -- same policy as test_voice_clone_workflow).
MAX_NEW_TOKENS = 512

TEXT = "今天的天气真不错，适合去公园散步。"

_TITLE_MD = "# Qwen3 TTS Demo"
_BTN_LABEL = "Generate (生成)"
_DISCLAIMER_FRAG = "Disclaimer"


@pytest.fixture(scope="module")
def cv_model(gpu):
    """Load the CustomVoice model once for this module; unload on teardown."""
    m = loader.load("custom-voice")
    yield m
    loader.unload(m)


@pytest.fixture(scope="module")
def parity(cv_model):
    """Build the UNMODIFIED upstream demo once around our loaded object."""
    ckpt = str(models.local_dir("custom-voice"))
    # Exactly the call upstream main() makes after parsing plain defaults,
    # except we plumb the suite latency guardrail through the OFFICIAL
    # `--max-new-tokens` flag surface instead of reading sys.argv.
    demo = upstream_demo.build_demo(
        cv_model, ckpt=ckpt, gen_kwargs_default={"max_new_tokens": MAX_NEW_TOKENS}
    )
    return {"tts": cv_model, "demo": demo, "ckpt": ckpt}


def _generate_dependency(demo: gr.Blocks):
    """Locate the custom-voice handler inside Gradio's own event registry."""
    hits = [
        bf for bf in demo.fns.values()
        if getattr(bf.fn, "__name__", "") == "run_instruct"
    ]
    assert len(hits) == 1, (
        f"expected exactly one upstream run_instruct listener registered in "
        f"demo.fns, found {len(hits)} (上游 build_demo 未注册唯一 run_instruct 监听)"
    )
    return hits[0]


def _first_choice(comp) -> str | None:
    """First dropdown choice, tolerant of str vs (value,label) item shapes."""
    choices = getattr(comp, "choices", None) or []
    if not choices:
        return None
    first = choices[0]
    return first[0] if isinstance(first, (tuple, list)) else first


# ---------------------------------------------------------------------------
# Test A: construction evidence -- genuinely THEIR layout, our model object
# ---------------------------------------------------------------------------

def test_upstream_blocks_constructed(parity):
    """Stock build_demo constructs a Blocks graph bound to OUR loaded model."""
    demo, ckpt = parity["demo"], parity["ckpt"]

    assert isinstance(demo, gr.Blocks), (
        f"build_demo did not return gr.Blocks, got {type(demo)}"
    )
    n_blocks = len(demo.blocks)
    print(f"[parity] upstream custom_voice Blocks registers {n_blocks} components")
    assert n_blocks >= 15, (
        f"upstream layout unexpectedly small: {n_blocks} blocks "
        f"(18-ish expected from the empirical upstream graph; see docstring "
        f"for why the plan's '>20' floor was right-sized 模块数不符)"
    )

    # >=1 Button AND the exact upstream generate-button label among values.
    buttons = [b.value for b in demo.blocks.values() if isinstance(b, gr.Button)]
    assert buttons, "no gr.Button in the constructed graph"
    assert _BTN_LABEL in buttons, (
        f"'{_BTN_LABEL}' missing; buttons={buttons!r} "
        f"(找不到上游生成按钮，疑似非官方布局)"
    )

    # Their title markdown renders, carrying OUR local checkpoint path.
    mds = [str(b.value) for b in demo.blocks.values() if type(b).__name__ == "Markdown"]
    joined = "\n".join(mds)
    assert any(_TITLE_MD in m for m in mds), (
        "upstream '# Qwen3 TTS Demo' markdown missing (缺少官方标题 Markdown)"
    )
    assert any(f"`{ckpt}`" in m for m in mds), (
        "checkpoint line with OUR local weights dir missing -- would mean "
        "build_demo was called with some other ckpt (未显示本机权重路径)"
    )
    assert "`custom_voice`" in joined, (
        "model-type markdown not 'custom_voice' (模型类型标注错误)"
    )
    assert _DISCLAIMER_FRAG in joined, "official disclaimer markdown missing"

    # Gradio-side wiring fingerprints of THEIR registration calls.
    dep = _generate_dependency(demo)
    labels = [(type(c).__name__, c.label) for c in dep.inputs]
    out_labels = [(type(c).__name__, c.label) for c in dep.outputs]
    print(f"[parity] run_instruct inputs={labels} outputs={out_labels}")
    assert labels == [
        ("Textbox", "Text (待合成文本)"),
        ("Dropdown", "Language (语种)"),
        ("Dropdown", "Speaker (说话人)"),
        ("Textbox", "Instruction (Optional) (控制指令，可不输入)"),
    ], f"upstream input signature changed/mismatched: {labels}"
    assert out_labels == [
        ("Audio", "Output Audio (合成结果)"),
        ("Textbox", "Status (状态)"),
    ], f"upstream output signature changed/mismatched: {out_labels}"


# ---------------------------------------------------------------------------
# Test B: closure execution THROUGH Gradio's own event registry
# ---------------------------------------------------------------------------

def test_official_closure_executes_via_gradio_fns(parity):
    """Invoke the registry-stored run_instruct closure -- validation branch.

    Uses the official empty-text guard so no GPU synthesis is paid here;
    a faithful (None, status) pair back proves the very callable Gradio would
    dispatch on a real click runs fine against our loader-produced object.
    """
    dep = _generate_dependency(parity["demo"])
    result = dep.fn("", "", "", "")
    assert result == (None, "Text is required (必须填写文本)."), (
        f"unexpected guard outcome from official closure: {result!r}"
    )
    print("[parity] run_instruct executed via demo.fns -> official empty-text guard OK")


# ---------------------------------------------------------------------------
# Test C: ONE real wav produced through the upstream callback, sane, timed
# ---------------------------------------------------------------------------

def test_real_synthesis_through_upstream_callback(parity):
    """End-to-end: click-equivalent invocation renders one sane waveform + RTF."""
    demo = parity["demo"]
    dep = _generate_dependency(demo)

    _, lang_dd, spk_dd, _ = dep.inputs
    language = _first_choice(lang_dd)
    speaker = _first_choice(spk_dd)
    assert language and speaker, (
        f"dropdowns had no usable choices: lang={language!r} spk={speaker!r} "
        f"(元数据来自我们加载的官方对象，不应为空)"
    )
    print(f"[parity] invoking run_instruct({TEXT!r}, {language!r}, {speaker!r}, '')")

    t0 = time.perf_counter()
    out, status = dep.fn(TEXT, language, speaker, "")
    wall = time.perf_counter() - t0

    assert status == "Finished. (生成完成)", (
        f"official callback reported failure: {status!r} "
        f"(上游回调返回失败状态)"
    )
    assert isinstance(out, tuple) and len(out) == 2, (
        f"expected upstream _wav_to_gradio_audio tuple, got {type(out)!r}: {out!r}"
    )
    sr, wav = out
    assert isinstance(sr, int) and sr > 0
    assert isinstance(wav, np.ndarray) and wav.dtype.kind == "f"
    audio_secs = wav.shape[-1] / float(sr)
    rtf = wall / audio_secs
    print(
        f"[timing] official run_instruct took={wall:.1f}s "
        f"audio={audio_secs:.2f}s sr={sr} rtf={rtf:.3f}"
    )
    testing.assert_wav_sane(wav, sr_expected=sr)
