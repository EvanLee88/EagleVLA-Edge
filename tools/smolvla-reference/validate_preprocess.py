#!/usr/bin/env python3
"""Compare C++ RGB preprocessing against LeRobot on the CPU."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from lerobot.policies.common.vla_utils import resize_with_pad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="built preprocess reference executable")
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    rng = np.random.default_rng(19)
    with tempfile.TemporaryDirectory(prefix="smolvla-preprocess-") as directory:
        source, output = Path(directory) / "image.u8", Path(directory) / "image.f32"
        for width, height in [(256, 256), (640, 360), (360, 640), (800, 777), (1, 512), (512, 1), (512, 512)]:
            image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            image.tofile(source)
            subprocess.run([str(executable), str(width), str(height), str(source), str(output)], check=True)
            image_pt = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            expected = (resize_with_pad(image_pt, 512, 512, pad_value=0) * 2 - 1).squeeze(0).permute(1, 2, 0).numpy()
            actual = np.fromfile(output, dtype=np.float32).reshape(512, 512, 3)
            error = np.abs(actual - expected)
            print(f"{width}x{height}: max={error.max():.8g} mean={error.mean():.8g}")
            if not np.isfinite(actual).all() or error.max() > 1e-4:
                raise AssertionError("SmolVLA image preprocessing differs from LeRobot")


if __name__ == "__main__":
    main()
