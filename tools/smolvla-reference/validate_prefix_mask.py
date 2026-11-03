#!/usr/bin/env python3
"""Check prefix attention and position IDs against a LeRobot fixture."""
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
    with tempfile.TemporaryDirectory(prefix='smolvla-prefix-') as temp, safe_open(args.reference, framework='numpy') as fixture:
        inp, mask, positions = (Path(temp) / name for name in ('language.u8', 'mask.u8', 'positions.i32'))
        fixture.get_tensor('language_mask').astype(np.uint8).tofile(inp)
        subprocess.run([str(args.executable.resolve()), str(inp), str(mask), str(positions)], check=True)
        expected_mask = fixture.get_tensor('prefix_attention').reshape(-1)
        expected_positions = fixture.get_tensor('prefix_positions').reshape(-1)
        if not np.array_equal(np.fromfile(mask, dtype=np.uint8), expected_mask):
            raise AssertionError('prefix attention mask mismatch')
        if not np.array_equal(np.fromfile(positions, dtype=np.int32), expected_positions):
            raise AssertionError('prefix position IDs mismatch')
        print(f'prefix mask and positions match: {expected_positions.size} tokens')


if __name__ == '__main__':
    main()
