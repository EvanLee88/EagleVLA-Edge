#!/usr/bin/env python3
"""Compare full RGB-to-action CPU policy with a deterministic LeRobot fixture."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('executable',type=Path);p.add_argument('model',type=Path);p.add_argument('mmproj',type=Path)
    p.add_argument('reference',type=Path);p.add_argument('metadata',type=Path)
    args=p.parse_args()
    instruction=json.loads(args.metadata.read_text())['instruction']
    with tempfile.TemporaryDirectory(prefix='smolvla-policy-') as temp, safe_open(args.reference,framework='numpy') as f:
        images=[Path(temp)/f'image{i}' for i in range(3)];state=Path(temp)/'state';noise=Path(temp)/'noise';out=Path(temp)/'out'
        for i,path in enumerate(images):f.get_tensor(f'image.{i}.rgb').tofile(path)
        f.get_tensor('state').reshape(-1)[:6].tofile(state);f.get_tensor('noise').tofile(noise)
        subprocess.run([str(args.executable.resolve()),str(args.model.resolve()),str(args.mmproj.resolve()),*map(str,images),str(state),str(noise),instruction,str(out)],check=True)
        actual=np.fromfile(out,np.float32).astype(np.float64);expected=f.get_tensor('action.normalized').reshape(-1).astype(np.float64)
        if actual.shape!=expected.shape or not np.isfinite(actual).all():raise AssertionError('invalid action output')
        diff=np.abs(actual-expected);cosine=np.dot(actual,expected)/(np.linalg.norm(actual)*np.linalg.norm(expected))
        print(f'max={diff.max():.8g} mean={diff.mean():.8g} cosine={cosine:.9f}')
        if diff.max()>0.05 or cosine<0.999:raise AssertionError('policy differs from LeRobot')


if __name__=='__main__':main()
