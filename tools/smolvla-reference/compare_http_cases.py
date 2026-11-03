#!/usr/bin/env python3
"""Compare SmolVLA actions from the foreground HTTP server and LeRobot on CPU.

Starts one local server using the README foreground options, with a fixed-noise
file rewritten between requests. The server reads that file on every inference.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.common.vla_utils import resize_with_pad
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from compare_cases import make_cases, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--mmproj', type=Path, required=True)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=18975)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1 or not 1 <= args.port <= 65535:
        p.error('threads and port must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.output.resolve()
    noise_path = root / 'active-noise.f32'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    torch.set_num_threads(args.threads)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = 'cpu'; cfg.load_vlm_weights = False; cfg.vlm_model_name = str(args.backbone.resolve())
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, config=cfg, local_files_only=True, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    env = os.environ.copy()
    env['PI_MODEL'] = 'auto'
    env['SMOLVLA_ACTION_NOISE_BIN'] = str(noise_path)
    command = [str(args.server.resolve()), '-m', str(args.model.resolve()), '--mmproj', str(args.mmproj.resolve()),
               '-ngl', '0', '--host', '127.0.0.1', '--port', str(args.port), '-np', '1', '-c', '1024',
               '-t', str(args.threads), '-tb', str(args.threads)]
    base = f'http://127.0.0.1:{args.port}'
    def request(method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)
    results, arrays = [], {}
    with (root / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited with {proc.returncode}; inspect {root / "server.log"}')
                try:
                    if request('GET', '/health').get('status') == 'ok':
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('server startup timed out')
                time.sleep(0.5)
            for name, case in make_cases():
                directory = root / name
                directory.mkdir(exist_ok=True)
                prepared, image_paths = [], []
                for i, image in enumerate(case['images']):
                    path = directory / f'camera{i}.png'
                    Image.fromarray(image).save(path)
                    image_paths.append(path)
                    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
                    prepared.append(resize_with_pad(tensor, 512, 512, pad_value=0) * 2 - 1)
                case['state'].tofile(directory / 'state.f32')
                case['noise'].tofile(directory / 'noise.f32')
                # Update only between synchronous calls, never while inference is running.
                case['noise'].tofile(noise_path)
                text = case['text'] if case['text'].endswith('\n') else case['text'] + '\n'
                language = tokenizer(text, padding='max_length', truncation=True, max_length=48, return_tensors='pt')
                state = torch.zeros((1, 32)); state[0, :6] = torch.from_numpy(case['state'])
                start = time.monotonic()
                with torch.inference_mode():
                    reference = policy.model.sample_actions(prepared, [torch.ones(1, dtype=torch.bool)] * 3,
                        language['input_ids'], language['attention_mask'].bool(), state,
                        noise=torch.from_numpy(case['noise'].copy()))[:, :, :6].float().numpy().reshape(50, 6)
                ref_time = time.monotonic() - start
                request('POST', '/foreground/reset', {})
                for path in image_paths:
                    request('POST', '/foreground/image', {'path': str(path)})
                request('PUT', '/foreground/state', {'state': case['state'].tolist()})
                start = time.monotonic()
                response = request('POST', '/foreground/infer', {'text': case['text']})
                edge_time = time.monotonic() - start
                if response.get('model_family') != 'smolvla' or response.get('action_normalized') is not True:
                    raise ValueError(f'{name}: unexpected model family or action scale')
                actual = np.asarray(response['action_final'], dtype=np.float32)
                if actual.shape != (50, 6) or not np.isfinite(actual).all() or not np.isfinite(reference).all():
                    raise ValueError(f'{name}: invalid action output')
                session = request('GET', '/foreground/session')
                if session['pending_bitmaps'] != 0:
                    raise ValueError(f'{name}: images were not consumed')
                (directory / 'response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2)+'\n')
                result = {'case': name, 'instruction': case['text'], 'normalized_state': case['state'].tolist(),
                          'reference_seconds': ref_time, 'http_infer_seconds': edge_time, **metrics(reference, actual)}
                results.append(result)
                arrays[name + '_lerobot'] = reference
                arrays[name + '_edge_http'] = actual
                print(f"{name}: max={result['max_abs']:.6g} MAE={result['mean_abs']:.6g} RMSE={result['rmse']:.6g} cosine={result['cosine']:.9f}", flush=True)
        finally:
            proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
    np.savez_compressed(root / 'actions.npz', **arrays)
    report = {'device': 'cpu', 'edge_transport': 'foreground HTTP', 'action_space': 'normalized',
              'seed': 20261030, 'noise': 'same 50x32 float32 per paired case', 'threads': args.threads,
              'server_command': command, 'torch_version': torch.__version__, 'platform': platform.platform(), 'cases': results}
    (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
