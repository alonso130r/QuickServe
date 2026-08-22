#!/usr/bin/env python3
"""Download a single Qwen3.5-2B GGUF quantization to the Hugging Face cache."""

import argparse
import pathlib
import sys


REPO_ID = "unsloth/Qwen3.5-2B-GGUF"
DEFAULT_QUANT = "Q4_K_M"


def select_model_file(files, quant):
    expected = f"Qwen3.5-2B-{quant}.gguf"
    matches = [name for name in files if pathlib.PurePosixPath(name).name == expected]
    if not matches:
        raise ValueError(f"No file named {expected!r} was found in {REPO_ID}")
    if len(matches) > 1:
        paths = ", ".join(matches)
        raise ValueError(f"Found multiple files named {expected!r}: {paths}")
    return matches[0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cache one Qwen3.5-2B GGUF quantization and print its path."
    )
    parser.add_argument("--quant", default=DEFAULT_QUANT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is required; install it with "
            "'python3 -m pip install huggingface_hub'",
            file=sys.stderr,
        )
        return 1

    try:
        filename = select_model_file(HfApi().list_repo_files(REPO_ID), args.quant)
        path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    except Exception as error:
        print(f"Unable to fetch {REPO_ID}: {error}", file=sys.stderr)
        return 1

    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
