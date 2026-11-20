#!/usr/bin/env python3
"""Upload and verify the pinned CUDA 12.8 wheel Release assets."""

from pathlib import Path
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.with_name("torch-cu118-cp311-linux_x86_64")))
    from publish import main

    sys.exit(main(root, release_tag="deps-torch-cu128-cp311-linux-x86_64-v1", allow_draft=True))
