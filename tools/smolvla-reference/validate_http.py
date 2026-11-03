#!/usr/bin/env python3
"""Exercise existing foreground routes with the deterministic SmolVLA fixture."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request
import numpy as np
from PIL import Image
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('url');p.add_argument('reference',type=Path);p.add_argument('metadata',type=Path);p.add_argument('directory',type=Path)
    args=p.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
    def request(method,path,data=None,expect=200):
        body=None if data is None else json.dumps(data).encode()
        req=urllib.request.Request(args.url.rstrip('/')+path,data=body,method=method,headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=300) as response: status,response_body=response.status,json.load(response)
        except urllib.error.HTTPError as e: status,response_body=e.code,json.load(e)
        if status!=expect: raise AssertionError(f'{path}: expected {expect}, got {status}: {response_body}')
        return response_body
    with safe_open(args.reference,framework='numpy') as fixture:
        request('POST','/foreground/reset',{})
        request('PUT','/foreground/state',{'state':[0,1]},400)
        state=fixture.get_tensor('state').reshape(-1)[:6]
        request('PUT','/foreground/state',{'state':state.tolist()})
        request('PUT','/foreground/state',{'state':','.join(str(float(x)) for x in state)})
        request('PUT','/foreground/state',{'state':'0,0,0,0,0,invalid'},400)
        for i in range(3):
            path=(args.directory/f'camera{i}.png').resolve();Image.fromarray(fixture.get_tensor(f'image.{i}.rgb')).save(path)
            request('POST','/foreground/image',{'path':str(path)})
        text=json.loads(args.metadata.read_text())['instruction']
        result=request('POST','/foreground/infer',{'text':text})
        actual=np.asarray(result['action_final'],dtype=np.float64);expected=fixture.get_tensor('action.normalized').reshape(50,6).astype(np.float64)
        if result['model_family']!='smolvla' or not result['action_normalized'] or actual.shape!=(50,6) or not np.isfinite(actual).all():raise AssertionError('invalid HTTP action response')
        error=np.abs(actual-expected);cosine=np.sum(actual*expected)/(np.linalg.norm(actual)*np.linalg.norm(expected))
        print(f'max={error.max():.8g} mean={error.mean():.8g} cosine={cosine:.9f}')
        if error.max()>0.05 or cosine<0.999:raise AssertionError('HTTP action differs from LeRobot')
        session=request('GET','/foreground/session')
        if session['pending_bitmaps']!=0 or not session['state_flag']:raise AssertionError('invalid session after inference')
        request('POST','/foreground/infer',{'text':text},400)
        request('POST','/foreground/reset',{})
        if request('GET','/foreground/session')['state_flag']:raise AssertionError('reset retained state')


if __name__=='__main__':main()
