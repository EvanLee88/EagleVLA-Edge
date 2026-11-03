"""Weight-free contract tests for the SmolVLA converters."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def validate():
    spec = importlib.util.spec_from_file_location("smolvla_hf_converter", ROOT / "convert_hf_to_gguf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_smolvla_cpu_config


@pytest.fixture
def policy():
    # Copied from smolvla_base/config.json; no checkpoint is needed.
    return {
        "type": "smolvla",
        "chunk_size": 50,
        "max_action_dim": 32,
        "max_state_dim": 32,
        "num_vlm_layers": 16,
        "num_steps": 10,
        "tokenizer_max_length": 48,
        "self_attn_every_n_layers": 2,
        "attention_mode": "cross_attn",
        "resize_imgs_with_padding": [512, 512],
        "min_period": 0.004,
        "max_period": 4.0,
        "empty_cameras": 0,
        "adapt_to_pi_aloha": False,
        "use_delta_joint_actions_aloha": False,
        "use_cache": True,
        "add_image_special_tokens": False,
        "pad_language_to": "max_length",
        "num_expert_layers": 0,
        "expert_width_multiplier": 0.75,
        "prefix_length": 0,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            **{f"observation.images.camera{i}": {"type": "VISUAL", "shape": [3, 256, 256]}
               for i in range(1, 4)},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }


def test_valid_cpu_config(validate, policy):
    original = copy.deepcopy(policy)
    validate(policy)
    assert policy == original


@pytest.mark.parametrize("key,value", [
    ("chunk_size", 49), ("max_action_dim", 31), ("max_state_dim", 31),
    ("num_vlm_layers", 15), ("num_steps", 9), ("tokenizer_max_length", 47),
    ("self_attn_every_n_layers", 1), ("attention_mode", "self_attn"),
    ("resize_imgs_with_padding", [256, 256]), ("min_period", 0.005),
    ("max_period", 3.0),
])
def test_rejects_unsupported_cpu_parameters(validate, policy, key, value):
    policy[key] = value
    with pytest.raises(ValueError):
        validate(policy)


@pytest.mark.parametrize("key", [
    "chunk_size", "max_action_dim", "max_state_dim", "num_vlm_layers", "num_steps",
    "tokenizer_max_length", "self_attn_every_n_layers", "attention_mode",
    "resize_imgs_with_padding", "min_period", "max_period",
])
def test_rejects_missing_cpu_parameters(validate, policy, key):
    del policy[key]
    with pytest.raises((ValueError, KeyError)):
        validate(policy)


@pytest.mark.parametrize("mutation", [
    "missing_camera", "extra_camera", "camera_shape", "camera_type",
    "missing_state", "state_shape", "state_type", "missing_action", "action_shape", "action_type",
])
def test_rejects_unsupported_features(validate, policy, mutation):
    inputs, outputs = policy["input_features"], policy["output_features"]
    camera = "observation.images.camera1"
    if mutation == "missing_camera":
        del inputs[camera]
    elif mutation == "extra_camera":
        inputs["observation.images.camera4"] = copy.deepcopy(inputs[camera])
    elif mutation == "camera_shape":
        inputs[camera]["shape"] = [3, 512, 512]
    elif mutation == "camera_type":
        inputs[camera]["type"] = "STATE"
    elif mutation == "missing_state":
        del inputs["observation.state"]
    elif mutation == "state_shape":
        inputs["observation.state"]["shape"] = [7]
    elif mutation == "state_type":
        inputs["observation.state"]["type"] = "VISUAL"
    elif mutation == "missing_action":
        del outputs["action"]
    elif mutation == "action_shape":
        outputs["action"]["shape"] = [7]
    elif mutation == "action_type":
        outputs["action"]["type"] = "STATE"
    with pytest.raises((ValueError, KeyError)):
        validate(policy)

