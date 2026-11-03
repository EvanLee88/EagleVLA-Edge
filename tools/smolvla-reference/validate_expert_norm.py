#!/usr/bin/env python3
"""Compare expert RMS normalization with LeRobot's recorded output."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('executable', type=Path)
    p.add_argument('model', type=Path)
    p.add_argument('reference', type=Path)
    p.add_argument('--max-abs', type=float, default=0.01)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='smolvla-norm-') as temp, safe_open(args.reference, framework='numpy') as fixture:
        inp, out = Path(temp) / 'input.f32', Path(temp) / 'output.f32'
        for step in range(10):
            fixture.get_tensor(f'step.{step}.expert_norm_input').tofile(inp)
            subprocess.run([str(args.executable.resolve()), str(args.model.resolve()), str(inp), str(out)], check=True)
            expected = fixture.get_tensor(f'step.{step}.expert_output').reshape(-1)
            actual = np.fromfile(out, dtype=np.float32)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise AssertionError(f'step {step}: invalid output')
            error = np.abs(actual - expected)
            print(f'step {step}: max={error.max():.8g} mean={error.mean():.8g}')
            if error.max() > args.max_abs:
                raise AssertionError(f'step {step}: RMS norm differs from LeRobot')


if __name__ == '__main__':
    main()
