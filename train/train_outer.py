#!/usr/bin/env python3
"""Unified outer-loop trainer for RecursiveMAS.

Pick the collaboration style with --style; everything after it is forwarded to that style's
outer trainer unchanged. The recursive gradient path through the (frozen) inner adapters is
ALWAYS preserved, so the outer RecursiveLink modules are trained correctly across rounds.

  python train_outer.py --style sequential_scaled  --agent1_model_name_or_path ...  ...
  python train_outer.py --style mixture            --agent1_model_name_or_path ...  ...
  python train_outer.py --style distillation       --expert_model_name_or_path ...  ...
  python train_outer.py --style deliberation       --reflector_model_name_or_path ... ...

Run `python train_outer.py --style <style> --help` to see that style's full argument list.
"""
import argparse
import sys

# Style -> outer family. sequential_light and sequential_scaled both use the sequential family
# (same recursive loop + roles); they differ only in which base models you pass.
STYLE_TO_FAMILY = {
    "sequential_light": "sequential",
    "sequential_scaled": "sequential",
    "mixture": "hie",
    "distillation": "distill",
    "deliberation": "deliberation",
}


def _family_main(family):
    if family == "sequential":
        from outer.sequential import main
    elif family == "hie":
        from outer.mixture import main
    elif family == "distill":
        from outer.distillation import main
    elif family == "deliberation":
        from outer.deliberation import main
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown outer family: {family}")
    return main


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False, description=__doc__)
    pre.add_argument("--style", required=True, choices=sorted(STYLE_TO_FAMILY))
    known, rest = pre.parse_known_args()

    family = STYLE_TO_FAMILY[known.style]
    family_main = _family_main(family)
    try:  # keep training output minimal: suppress datasets .map() progress bars
        import datasets as _datasets
        _datasets.disable_progress_bars()
    except Exception:
        pass
    family_main(rest)


if __name__ == "__main__":
    main()
