import argparse

from .env import rocm_check


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qwen3-tts-rocm-check",
                                 description="Diagnose ROCm GPU readiness (环境自检)")
    ap.add_argument("-q", "--quiet", action="store_true")
    ns = ap.parse_args(argv)
    report = rocm_check(verbose=not ns.quiet)
    return 0 if not report.errors else 1
