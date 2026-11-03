#!/usr/bin/env python3
"""Compare CPU VLM prefix caches and subsequent actions against LeRobot."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('prefix_executable',type=Path);p.add_argument('sample_executable',type=Path)
    p.add_argument('model',type=Path);p.add_argument('reference',type=Path)
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='smolvla-prefix-') as temp, safe_open(args.reference,framework='numpy') as f:
        prefix,mask,pos,keys,values,noise,padding,out=(Path(temp)/n for n in ('prefix','mask','pos','keys','values','noise','padding','out'))
        f.get_tensor('prefix_embedding').astype(np.float32).tofile(prefix)
        f.get_tensor('prefix_attention').astype(np.uint8).tofile(mask)
        f.get_tensor('prefix_positions').astype(np.int32).tofile(pos)
        f.get_tensor('prefix_padding').astype(np.uint8).tofile(padding)
        f.get_tensor('noise').tofile(noise)
        n=f.get_tensor('prefix_positions').size
        subprocess.run([str(args.prefix_executable.resolve()),str(args.model.resolve()),str(n),str(prefix),str(mask),str(pos),str(keys),str(values)],check=True)
        for kind,path in (('key',keys),('value',values)):
            actual=np.fromfile(path,np.float32).reshape(16,n,5,64).astype(np.float64)
            for layer in range(16):
                expected=f.get_tensor(f'cache.{layer}.{kind}').transpose(0,2,1,3).reshape(n,5,64).astype(np.float64)
                a=actual[layer]; diff=np.abs(a-expected); cosine=np.sum(a*expected)/(np.linalg.norm(a)*np.linalg.norm(expected))
                print(f'{kind} {layer}: max={diff.max():.8g} mean={diff.mean():.8g} cosine={cosine:.9f}')
                if diff.max()>0.25 or cosine<0.9999: raise AssertionError('prefix cache differs from LeRobot')
        subprocess.run([str(args.sample_executable.resolve()),str(args.model.resolve()),str(noise),str(keys),str(values),str(padding),str(n),str(out)],check=True)
        actual=np.fromfile(out,np.float32).reshape(50,32)[:,:6].astype(np.float64)
        expected=f.get_tensor('action.normalized').reshape(50,6).astype(np.float64)
        diff=np.abs(actual-expected); cosine=np.sum(actual*expected)/(np.linalg.norm(actual)*np.linalg.norm(expected))
        print(f'action: max={diff.max():.8g} mean={diff.mean():.8g} cosine={cosine:.9f}')
        if diff.max()>0.05 or cosine<0.999: raise AssertionError('sample differs from LeRobot')


if __name__=='__main__': main()
