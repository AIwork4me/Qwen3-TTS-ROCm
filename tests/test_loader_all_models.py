# tests/test_loader_all_models.py
"""Task 10: GPU integration -- six-model load/unload smoke (first real loads).

Loads every downloaded TTS repo through :func:`qwen3_tts_rocm.loader.load` on
the Radeon 8060S (gfx1151) plus the speech tokenizer via the official
``Qwen3TTSTokenizer.from_pretrained``, asserting the returned object is the
native official model sitting on CUDA.  Each parametrized case performs a
fresh load + unload (NOT the session cache) so memory behaviour is observed
per model; after each load a ``[vram]`` line with free/used totals and the
allocator peak is printed to stdout for evidence (tee'd to
``evidence/load-smoke.txt``).

Known-plan deviation (ruling recorded in task-10 brief preflight): the dead
``HEAVY = {"tokenizer"}`` module constant from the original snippet was
dropped entirely.
"""

import pytest

from qwen3_tts_rocm import loader, models, testing  # noqa: F401 (testing: contract parity)

pytestmark = [pytest.mark.gpu, pytest.mark.requires_download]


def _vram_line(tag: str) -> None:
    """Print one evidence line: free/used VRAM + torch allocator peak (MiB)."""
    import torch

    free, total = torch.cuda.mem_get_info(0)
    peak = torch.cuda.max_memory_allocated(0) / 2**20
    print(
        f"[vram] {tag}: used={(total - free) / 2**20:.1f} MiB "
        f"free={free / 2**20:.1f} MiB total={total / 2**20:.1f} MiB "
        f"allocator_peak={peak:.1f} MiB"
    )


@pytest.mark.parametrize("alias", ["custom-voice", "voice-design", "base", "custom-voice-0.6b", "base-0.6b"])
def test_load_smoke_each_tts_model(alias, gpu):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    m = loader.load(alias)
    assert m.model.device.type == "cuda"
    _vram_line(f"after load {alias}")
    loader.unload(m)


def test_tokenizer_load(gpu):
    import torch
    from qwen_tts import Qwen3TTSTokenizer

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    tok = Qwen3TTSTokenizer.from_pretrained(str(models.local_dir("tokenizer")), device_map="cuda:0", dtype="bfloat16")
    assert tok.get_output_sample_rate() > 0
    _vram_line("after load tokenizer")
    loader.unload(tok)  # best-effort teardown (.model attr exists -> weights freed)
