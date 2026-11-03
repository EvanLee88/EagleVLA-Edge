#!/usr/bin/env python3
"""Compare LeRobot and EagleVLA-Edge CPU actions on reproducible synthetic cases.

All state inputs and actions are normalized. The same image bytes, tokenized
instruction, state and initial noise are used for both implementations.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.common.vla_utils import resize_with_pad
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def metrics(reference, actual):
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    delta = b - a
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return {
        'max_abs': float(np.max(np.abs(delta))),
        'mean_abs': float(np.mean(np.abs(delta))),
        'rmse': float(np.sqrt(np.mean(delta * delta))),
        'relative_l2': float(np.linalg.norm(delta) / np.linalg.norm(a)) if np.linalg.norm(a) else None,
        'cosine': float(np.sum(a * b) / norm) if norm else None,
        'per_dimension_mean_abs': np.mean(np.abs(delta), axis=0).tolist(),
        'per_dimension_max_abs': np.max(np.abs(delta), axis=0).tolist(),
    }


def make_cases():
    rng = np.random.default_rng(20261030)
    images = rng.integers(0, 256, (3, 256, 256, 3), dtype=np.uint8)
    noise = rng.standard_normal((1, 50, 32)).astype(np.float32)
    state = np.array([0.125, -0.25, 0.375, -0.5, 0.625, -0.75], dtype=np.float32)
    base = {'images': images, 'noise': noise, 'state': state,
            'text': 'Pick up the red block and place it in the tray.'}
    varied_images = images.copy()
    varied_images[1] = 255 - varied_images[1]
    other_state = np.array([-0.8, 0.5, -0.3, 0.1, 0.2, 0.9], dtype=np.float32)
    black_images = np.zeros_like(images)
    other_noise = rng.standard_normal((1, 50, 32)).astype(np.float32)
    return [
        ('baseline', base),
        ('instruction', {**base, 'text': 'Move the blue object to the left and release it.'}),
        ('state', {**base, 'state': other_state}),
        ('camera', {**base, 'images': varied_images}),
        ('noise', {**base, 'noise': other_noise}),
        ('black_long_instruction', {**base, 'images': black_images, 'state': np.zeros(6, np.float32),
            'text': 'Carefully pick up the red block and place it into the tray without touching any other object. ' * 4}),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--mmproj', type=Path, required=True)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads != 4:
        p.error('the reference executable currently uses four CPU threads')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = 'cpu'; cfg.load_vlm_weights = False; cfg.vlm_model_name = str(args.backbone.resolve())
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, config=cfg, local_files_only=True, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    results, arrays = [], {}
    for name, case in make_cases():
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        image_paths = []
        prepared = []
        for index, image in enumerate(case['images']):
            path = directory / f'image{index}.rgb'; image.tofile(path); image_paths.append(str(path.resolve()))
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
            prepared.append(resize_with_pad(tensor, 512, 512, pad_value=0) * 2 - 1)
        (directory / 'state.f32').write_bytes(case['state'].tobytes())
        (directory / 'noise.f32').write_bytes(case['noise'].tobytes())
        text = case['text'] if case['text'].endswith('\n') else case['text'] + '\n'
        language = tokenizer(text, padding='max_length', truncation=True, max_length=48, return_tensors='pt')
        state = torch.zeros((1, 32)); state[0, :6] = torch.from_numpy(case['state'])
        start = time.monotonic()
        with torch.inference_mode():
            reference = policy.model.sample_actions(prepared, [torch.ones(1, dtype=torch.bool)] * 3,
                language['input_ids'], language['attention_mask'].bool(), state,
                noise=torch.from_numpy(case['noise'].copy()))[:, :, :6].float().numpy().reshape(50, 6)
        ref_time = time.monotonic() - start
        output = directory / 'edge.f32'
        command = [str(args.executable.resolve()), str(args.model.resolve()), str(args.mmproj.resolve()),
                   *image_paths, str((directory/'state.f32').resolve()), str((directory/'noise.f32').resolve()),
                   case['text'], str(output.resolve())]
        start = time.monotonic()
        with (directory/'edge.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        edge_time = time.monotonic() - start
        actual = np.fromfile(output, np.float32)
        if actual.size != 300 or not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError(f'{name}: invalid action output')
        actual = actual.reshape(50, 6)
        result = {'case': name, 'instruction': case['text'], 'normalized_state': case['state'].tolist(),
                  'reference_seconds': ref_time, 'edge_seconds_including_load': edge_time, **metrics(reference, actual)}
        results.append(result)
        arrays[name + '_lerobot'] = reference
        arrays[name + '_edge'] = actual
        print(f"{name}: max={result['max_abs']:.6g} MAE={result['mean_abs']:.6g} RMSE={result['rmse']:.6g} cosine={result['cosine']:.9f}", flush=True)
    np.savez_compressed(args.output / 'actions.npz', **arrays)
    report = {'device': 'cpu', 'action_space': 'normalized', 'seed': 20261030,
              'noise': 'identical fixed float32 noise per paired case', 'threads': args.threads,
              'torch_version': torch.__version__, 'platform': platform.platform(),
              'cases': results}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
