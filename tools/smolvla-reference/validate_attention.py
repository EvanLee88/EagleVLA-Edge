#!/usr/bin/env python3
"""Compare GGML grouped-query attention with LeRobot's eager formula."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(182)
    with tempfile.TemporaryDirectory(prefix='smolvla-attn-') as temp:
        paths = [Path(temp) / n for n in ('q', 'k', 'v', 'mask', 'out')]
        for nq, nk in [(1, 1), (50, 241), (50, 291), (241, 241)]:
            q = rng.standard_normal((nq, 15, 64)).astype(np.float32)
            k = rng.standard_normal((nk, 5, 64)).astype(np.float32)
            v = rng.standard_normal((nk, 5, 64)).astype(np.float32)
            mask = (rng.random((nq, nk)) > 0.2).astype(np.uint8)
            mask[0] = 0  # LeRobot's completely masked query semantics
            for x, path in zip((q, k, v, mask), paths): x.tofile(path)
            subprocess.run([str(args.executable.resolve()), str(nq), str(nk), *map(str, paths)], check=True)
            qt = torch.from_numpy(q).permute(1, 0, 2)
            kt = torch.from_numpy(k).repeat_interleave(3, dim=1).permute(1, 0, 2)
            vt = torch.from_numpy(v).repeat_interleave(3, dim=1).permute(1, 0, 2)
            scores = (qt @ kt.transpose(1, 2)) * (64 ** -0.5)
            scores = torch.where(torch.from_numpy(mask).bool()[None], scores, torch.finfo(torch.float32).min)
            expected = (scores.softmax(-1) @ vt).permute(1, 0, 2).numpy().reshape(-1)
            actual = np.fromfile(paths[-1], np.float32)
            if actual.shape != expected.shape or not np.isfinite(actual).all(): raise AssertionError('invalid attention output')
            delta = np.abs(actual - expected)
            print(f'{nq}x{nk}: max={delta.max():.8g} mean={delta.mean():.8g}')
            if delta.max() > 2e-5: raise AssertionError('attention mismatch')


if __name__ == '__main__':
    main()
