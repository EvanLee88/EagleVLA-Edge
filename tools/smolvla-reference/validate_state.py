#!/usr/bin/env python3
"""Compare GGML state projection with the state token in a LeRobot fixture."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--max-abs', type=float, default=0.01)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='smolvla-state-') as temp, safe_open(args.reference, framework='numpy') as fixture:
        inp, out = Path(temp) / 'state.f32', Path(temp) / 'embedding.f32'
        state = fixture.get_tensor('state')
        expected = fixture.get_tensor('prefix_embedding')[0, -1, :]
        if state.shape != (1, 32) or expected.shape != (960,):
            raise ValueError('unexpected fixture dimensions')
        state.tofile(inp)
        subprocess.run([str(args.executable.resolve()), str(args.model.resolve()), str(inp), str(out)], check=True)
        actual = np.fromfile(out, dtype=np.float32)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise AssertionError('invalid state projection output')
        error = np.abs(actual - expected)
        print(f'max={error.max():.8g} mean={error.mean():.8g}')
        if error.max() > args.max_abs:
            raise AssertionError(f'maximum error exceeds {args.max_abs}')


if __name__ == '__main__':
    main()
