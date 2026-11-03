#!/usr/bin/env python3
"""Compare 10-step CPU sampling with LeRobot, using reference prefix KV."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('executable',type=Path);p.add_argument('model',type=Path);p.add_argument('reference',type=Path)
    p.add_argument('--max-abs',type=float,default=0.05);p.add_argument('--min-cosine',type=float,default=0.999)
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='smolvla-sample-') as temp, safe_open(args.reference,framework='numpy') as fixture:
        noise,keys,values,mask,out=(Path(temp)/n for n in ('noise','keys','values','mask','out'))
        fixture.get_tensor('noise').tofile(noise)
        padding=fixture.get_tensor('prefix_padding').reshape(-1);padding.astype(np.uint8).tofile(mask)
        for name,path in (('key',keys),('value',values)):
            np.stack([fixture.get_tensor(f'cache.{i}.{name}').transpose(0,2,1,3) for i in range(16)]).tofile(path)
        subprocess.run([str(args.executable.resolve()),str(args.model.resolve()),str(noise),str(keys),str(values),str(mask),str(padding.size),str(out)],check=True)
        expected=fixture.get_tensor('action.normalized').reshape(50,6).astype(np.float64)
        result=np.fromfile(out,np.float32)
        if result.size != 50*32 or not np.isfinite(result).all(): raise AssertionError('invalid sampled actions')
        actual=result.reshape(50,32)[:,:6].astype(np.float64)
        error=np.abs(actual-expected); cosine=np.sum(actual*expected)/(np.linalg.norm(actual)*np.linalg.norm(expected))
        print(f'max={error.max():.8g} mean={error.mean():.8g} cosine={cosine:.9f}')
        if error.max()>args.max_abs or cosine<args.min_cosine: raise AssertionError('sample differs from LeRobot')


if __name__=='__main__': main()
