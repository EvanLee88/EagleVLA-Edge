#!/usr/bin/env python3
"""Validate expert Q/K/V projections against LeRobot's saved trajectory."""
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
    p.add_argument('--max-abs', type=float, default=0.1)
    args = p.parse_args()
    maximum = 0.0
    with tempfile.TemporaryDirectory(prefix='smolvla-qkv-') as temp, safe_open(args.reference, framework='numpy') as f:
        h, ki, vi, qo, ko, vo = [Path(temp) / n for n in ('hidden', 'key', 'value', 'qout', 'kout', 'vout')]
        for layer in range(16):
            for step in range(10):
                prefix = f'step.{step}.layer.{layer}.'
                f.get_tensor(prefix + 'hidden_input').tofile(h)
                if layer % 2:
                    keys, values = f.get_tensor(prefix+'k_proj.input'), f.get_tensor(prefix+'v_proj.input')
                else:
                    keys = values = np.zeros((1, 241, 320), np.float32)
                keys.tofile(ki); values.tofile(vi)
                subprocess.run([str(args.executable.resolve()), str(args.model.resolve()), str(layer), str(h), str(ki), str(vi), str(keys.shape[1]), str(qo), str(ko), str(vo)], check=True)
                for name, path in (('q_proj', qo), ('k_proj', ko), ('v_proj', vo)):
                    expected = f.get_tensor(prefix+name+'.output').reshape(-1)
                    actual = np.fromfile(path, np.float32)
                    if expected.shape != actual.shape or not np.isfinite(actual).all(): raise AssertionError('invalid output')
                    diff = np.abs(actual-expected)
                    maximum = max(maximum, float(diff.max()))
                    if diff.max() > args.max_abs: raise AssertionError(f'layer {layer} step {step} {name}: {diff.max()}')
            print(f'layer {layer}: passed')
    print(f'max absolute error: {maximum:.8g}')


if __name__ == '__main__':
    main()
