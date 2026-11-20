#!/usr/bin/env python3
"""Verify and install the CUDA 12.8 wheel set offline."""

from pathlib import Path
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.with_name("torch-cu118-cp311-linux_x86_64")))
    from install import main

    sys.exit(main(root=root))
