#!/usr/bin/env python3
"""Compare GGML expert split-half RoPE against LeRobot on CPU."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('executable', type=Path)
    args = p.parse_args()
    rng = np.random.default_rng(715)
    x = rng.standard_normal((1, 50, 15, 64)).astype(np.float32)
    positions = rng.integers(0, 300, (1, 50), dtype=np.int32)
    expected = apply_rope(torch.from_numpy(x), torch.from_numpy(positions)).numpy().reshape(-1)
    with tempfile.TemporaryDirectory(prefix='smolvla-rope-') as temp:
        inp, pos, out = (Path(temp) / name for name in ('input', 'pos', 'output'))
        x.tofile(inp)
        positions.tofile(pos)
        subprocess.run([str(args.executable.resolve()), str(inp), str(pos), str(out)], check=True)
        actual = np.fromfile(out, np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError('invalid RoPE output')
    error = np.abs(actual - expected)
    print(f'max={error.max():.8g} mean={error.mean():.8g}')
    if error.max() > 1e-4:
        raise AssertionError('expert RoPE differs from LeRobot')


if __name__ == '__main__':
    main()
