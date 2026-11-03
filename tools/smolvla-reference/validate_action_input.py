#!/usr/bin/env python3
"""Compare GGML action/time projection against all LeRobot denoising steps."""
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
    with tempfile.TemporaryDirectory(prefix='smolvla-action-') as temp, safe_open(args.reference, framework='numpy') as fixture:
        inp, out = Path(temp) / 'input.f32', Path(temp) / 'output.f32'
        for step in range(10):
            fixture.get_tensor(f'step.{step}.input').tofile(inp)
            subprocess.run([str(args.executable.resolve()), str(args.model.resolve()), str(inp), str(1 - step / 10), str(out)], check=True)
            reference = fixture.get_tensor(f'step.{step}.suffix').reshape(-1)
            actual = np.fromfile(out, np.float32)
            if actual.shape != reference.shape or not np.isfinite(actual).all():
                raise AssertionError(f'step {step}: invalid shape or values')
            diff = np.abs(actual - reference)
            print(f'step {step}: max={diff.max():.8g} mean={diff.mean():.8g}')
            if diff.max() > args.max_abs:
                raise AssertionError(f'step {step}: maximum error exceeds {args.max_abs}')


if __name__ == '__main__':
    main()
