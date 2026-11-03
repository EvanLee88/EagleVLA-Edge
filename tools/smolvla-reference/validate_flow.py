#!/usr/bin/env python3
"""Replay LeRobot's saved velocities through the C++ Euler integrator."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('reference', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='smolvla-flow-') as temp, safe_open(args.reference, framework='numpy') as f:
        noise, velocity, output = (Path(temp) / name for name in ('noise.f32', 'velocity.f32', 'output.f32'))
        f.get_tensor('noise').tofile(noise)
        np.stack([f.get_tensor(f'step.{i}.velocity') for i in range(10)]).tofile(velocity)
        subprocess.run([str(args.executable.resolve()), str(noise), str(velocity), str(output)], check=True)
        expected = f.get_tensor('action.padded_normalized').reshape(-1)
        actual = np.fromfile(output, np.float32)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise AssertionError('invalid integrated action')
        difference = np.abs(actual - expected)
        print(f'max={difference.max():.8g} mean={difference.mean():.8g}')
        if difference.max() > 1e-6:
            raise AssertionError('Euler integration differs from LeRobot')


if __name__ == '__main__':
    main()
