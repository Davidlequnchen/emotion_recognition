#!/usr/bin/env python3
"""Audit the CUDA 12.8 wheels and their offline installation order."""

from pathlib import Path
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.with_name("torch-cu118-cp311-linux_x86_64")))
    from audit import main

    sys.exit(main(root=root))
