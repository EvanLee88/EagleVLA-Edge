#!/usr/bin/env python3
"""Compare RGB-to-embedding SmolVLA CPU vision execution with LeRobot."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.common.vla_utils import resize_with_pad
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("mmproj", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = "cpu"
    # The checkpoint already contains the VLM weights; do not download them again.
    config.load_vlm_weights = False
    model = SmolVLAPolicy.from_pretrained(args.checkpoint, config=config, local_files_only=True, strict=True)
    image = np.random.default_rng(31).integers(0, 256, (360, 640, 3), dtype=np.uint8)
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.inference_mode():
        expected = model.model.vlm_with_expert.embed_image(resize_with_pad(x, 512, 512, pad_value=0) * 2 - 1)
    expected = expected.float().numpy().reshape(-1).astype(np.float64)
    with tempfile.TemporaryDirectory(prefix="smolvla-vision-") as directory:
        source, output = Path(directory) / "image.rgb", Path(directory) / "embedding.f32"
        image.tofile(source)
        subprocess.run([str(args.executable.resolve()), str(args.mmproj.resolve()), "640", "360", str(source), str(output)], check=True)
        actual = np.fromfile(output, dtype=np.float32).astype(np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError(f"invalid output shape or values: {actual.shape}, expected {expected.shape}")
    error = np.abs(actual - expected)
    cosine = np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
    print(f"count={actual.size} max_abs={error.max():.8g} mean_abs={error.mean():.8g} cosine={cosine:.9f}")
    if error.max() > 0.05 or cosine < 0.99999:
        raise AssertionError("SmolVLA vision output differs from LeRobot")


if __name__ == "__main__":
    main()
