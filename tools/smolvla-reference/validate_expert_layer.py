#!/usr/bin/env python3
"""Compare one GGML expert layer at a time using LeRobot prefix caches."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('executable',type=Path); p.add_argument('model',type=Path); p.add_argument('reference',type=Path)
    p.add_argument('--max-abs',type=float,default=0.25)
    args=p.parse_args(); largest=0.0; error_sum=0.0; count=0
    with tempfile.TemporaryDirectory(prefix='smolvla-layer-') as temp, safe_open(args.reference,framework='numpy') as fixture:
        h,k,v,mask,out=[Path(temp)/n for n in ('h','k','v','mask','out')]
        padding=fixture.get_tensor('prefix_padding').reshape(-1); padding.astype(np.uint8).tofile(mask)
        for layer in range(16):
            fixture.get_tensor(f'cache.{layer}.key').transpose(0,2,1,3).copy().tofile(k)
            fixture.get_tensor(f'cache.{layer}.value').transpose(0,2,1,3).copy().tofile(v)
            for step in range(10):
                prefix=f'step.{step}.layer.{layer}.'
                fixture.get_tensor(prefix+'hidden_input').tofile(h)
                subprocess.run([str(args.executable.resolve()),str(args.model.resolve()),str(layer),str(h),str(k),str(v),str(mask),str(padding.size),str(out)],check=True)
                expected=fixture.get_tensor(prefix+'layer_output').reshape(-1); actual=np.fromfile(out,np.float32)
                if actual.shape!=expected.shape or not np.isfinite(actual).all(): raise AssertionError('invalid layer output')
                diff=np.abs(actual-expected); largest=max(largest,float(diff.max())); error_sum+=float(diff.sum()); count+=diff.size
                if diff.max()>args.max_abs: raise AssertionError(f'layer {layer} step {step}: {diff.max()}')
            print(f'layer {layer}: passed')
    print(f'max={largest:.8g} mean={error_sum/count:.8g}')


if __name__=='__main__': main()
