#!/usr/bin/env python3
"""Compare the full CPU expert against LeRobot, using reference prefix caches."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('executable', type=Path); p.add_argument('model', type=Path); p.add_argument('reference', type=Path)
    p.add_argument('--max-abs', type=float, default=0.5); p.add_argument('--min-cosine', type=float, default=0.999)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='smolvla-expert-') as temp, safe_open(args.reference, framework='numpy') as fixture:
        suffix, keys, values, mask, output = (Path(temp) / n for n in ('suffix','keys','values','mask','output'))
        padding = fixture.get_tensor('prefix_padding').reshape(-1); padding.astype(np.uint8).tofile(mask)
        for typ, path in (('key',keys),('value',values)):
            np.stack([fixture.get_tensor(f'cache.{i}.{typ}').transpose(0,2,1,3) for i in range(16)]).tofile(path)
        for step in range(10):
            fixture.get_tensor(f'step.{step}.suffix').tofile(suffix)
            subprocess.run([str(args.executable.resolve()),str(args.model.resolve()),str(suffix),str(keys),str(values),str(mask),str(padding.size),str(output)],check=True)
            expected=fixture.get_tensor(f'step.{step}.expert_output').reshape(-1).astype(np.float64)
            actual=np.fromfile(output,np.float32).astype(np.float64)
            if actual.shape != expected.shape or not np.isfinite(actual).all(): raise AssertionError('invalid expert output')
            error=np.abs(expected-actual); cosine=np.dot(expected,actual)/(np.linalg.norm(expected)*np.linalg.norm(actual))
            print(f'step {step}: max={error.max():.8g} mean={error.mean():.8g} cosine={cosine:.9f}',flush=True)
            if error.max()>args.max_abs or cosine<args.min_cosine: raise AssertionError('expert output differs from reference')


if __name__=='__main__': main()
