#!/usr/bin/env python3
"""Compare GGML expert MLP layers against LeRobot's recorded trajectory."""
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
    parser.add_argument('--max-abs', type=float, default=0.1)
    args = parser.parse_args()
    largest = 0.0
    with tempfile.TemporaryDirectory(prefix='smolvla-mlp-') as temp, safe_open(args.reference, framework='numpy') as fixture:
        inp, out = Path(temp) / 'input.f32', Path(temp) / 'output.f32'
        for layer in range(16):
            for step in range(10):
                fixture.get_tensor(f'step.{step}.layer.{layer}.mlp_input').tofile(inp)
                subprocess.run([str(args.executable.resolve()), str(args.model.resolve()), str(layer), str(inp), str(out)], check=True)
                expected = fixture.get_tensor(f'step.{step}.layer.{layer}.mlp_output').reshape(-1)
                actual = np.fromfile(out, dtype=np.float32)
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise AssertionError(f'layer {layer} step {step}: invalid output')
                error = np.abs(actual - expected)
                largest = max(largest, float(error.max()))
                if error.max() > args.max_abs:
                    raise AssertionError(f'layer {layer} step {step}: max={error.max()}')
            print(f'layer {layer}: passed')
    print(f'max absolute error: {largest:.8g}')


if __name__ == '__main__':
    main()
