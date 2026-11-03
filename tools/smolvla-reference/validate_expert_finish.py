#!/usr/bin/env python3
"""Compare expert attention output projection, residuals, RMSNorm and MLP."""
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
    p.add_argument('--max-abs', type=float, default=0.15)
    args = p.parse_args()
    largest, sum_error, count = 0.0, 0.0, 0
    with tempfile.TemporaryDirectory(prefix='smolvla-finish-') as temp, safe_open(args.reference, framework='numpy') as fixture:
        hidden, attention, out = (Path(temp) / name for name in ('hidden', 'attention', 'output'))
        for layer in range(16):
            for step in range(10):
                prefix = f'step.{step}.layer.{layer}.'
                fixture.get_tensor(prefix + 'hidden_input').tofile(hidden)
                fixture.get_tensor(prefix + 'attention_output').tofile(attention)
                subprocess.run([str(args.executable.resolve()), str(args.model.resolve()), str(layer), str(hidden), str(attention), str(out)], check=True)
                expected = fixture.get_tensor(prefix + 'layer_output').reshape(-1)
                actual = np.fromfile(out, dtype=np.float32)
                if actual.shape != expected.shape or not np.isfinite(actual).all(): raise AssertionError('invalid layer output')
                error = np.abs(actual - expected)
                largest = max(largest, float(error.max()))
                sum_error += float(error.sum())
                count += error.size
                if error.max() > args.max_abs: raise AssertionError(f'layer {layer}, step {step}: max {error.max()}')
            print(f'layer {layer}: passed')
    print(f'max={largest:.8g} mean={sum_error/count:.8g}')


if __name__ == '__main__':
    main()
