#!/usr/bin/env python3
"""Check action attention masks and positions against LeRobot's mask helper."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from safetensors import safe_open
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('executable', type=Path)
    p.add_argument('reference', type=Path)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='smolvla-suffix-') as temp, safe_open(args.reference, framework='numpy') as f:
        prefix = f.get_tensor('prefix_padding').reshape(-1)
        inp, self_file, cross_file, pos_file = (Path(temp) / n for n in ('prefix', 'self', 'cross', 'pos'))
        prefix.astype(np.uint8).tofile(inp)
        subprocess.run([str(args.executable.resolve()), str(inp), str(self_file), str(cross_file), str(pos_file)], check=True)
        prefix_t = torch.from_numpy(prefix.copy()).bool().reshape(1, -1)
        suffix_pad = torch.ones((1, 50), dtype=torch.bool)
        suffix_att = torch.ones((1, 50), dtype=torch.bool)
        suffix_mask = make_att_2d_masks(suffix_pad, suffix_att)
        full = torch.cat([prefix_t[:, None, :].expand(1, 50, -1), suffix_mask], dim=2).numpy().reshape(-1)
        cross = np.broadcast_to(prefix, (50, prefix.size)).reshape(-1)
        positions = (prefix_t.sum() + torch.cumsum(suffix_pad, dim=1) - 1).numpy().reshape(-1)
        assert np.array_equal(np.fromfile(self_file, np.uint8), full)
        assert np.array_equal(np.fromfile(cross_file, np.uint8), cross)
        assert np.array_equal(np.fromfile(pos_file, np.int32), positions)
        print(f'self/cross masks and positions match: {prefix.size} prefix, 50 suffix tokens')


if __name__ == '__main__':
    main()
