#!/usr/bin/env python3
"""Download or verify the CUDA 12.8 wheel set using the shared downloader."""

from pathlib import Path
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.with_name("torch-cu118-cp311-linux_x86_64")))
    from download import main

    sys.exit(main(root=root))
