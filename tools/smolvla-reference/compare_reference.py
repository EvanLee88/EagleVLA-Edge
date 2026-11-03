#!/usr/bin/env python3
"""Compare named CPU runtime tensors with a SmolVLA safetensors fixture."""
import argparse
import numpy as np
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--tensor', action='append', required=True, help='tensor name (repeatable)')
    parser.add_argument('--atol', type=float, default=0.05)
    parser.add_argument('--rtol', type=float, default=0.01)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error('tolerances must be nonnegative')
    ok = True
    with safe_open(args.reference, framework='numpy') as ref, safe_open(args.candidate, framework='numpy') as cand:
        for name in args.tensor:
            if name not in ref.keys() or name not in cand.keys():
                print(f'{name}: missing tensor')
                ok = False
                continue
            a, b = ref.get_tensor(name), cand.get_tensor(name)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                print(f'{name}: invalid shape or non-finite values ({a.shape}, {b.shape})')
                ok = False
                continue
            a, b = a.astype(np.float64), b.astype(np.float64)
            error = np.abs(a - b)
            passed = bool(np.all(error <= args.atol + args.rtol * np.abs(a)))
            print(f'{name}: max_abs={error.max():.8g} mean_abs={error.mean():.8g} passed={passed}')
            ok &= passed
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
