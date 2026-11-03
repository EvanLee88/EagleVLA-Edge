#!/usr/bin/env python3
"""Export a deterministic, normalized-input SmolVLA CPU inference fixture.

The synthetic state is already normalized. No postprocessor is applied: the
checkpoint does not provide unambiguous state/action statistics for this case.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.common.vla_utils import resize_with_pad
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--backbone', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--instruction', default='Pick up the red block and place it in the tray.')
    args = p.parse_args()
    if args.threads < 1:
        p.error('--threads must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = 'cpu'
    config.load_vlm_weights = False
    config.vlm_model_name = str(args.backbone.resolve())
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, config=config, local_files_only=True, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    instruction = args.instruction if args.instruction.endswith('\n') else args.instruction + '\n'
    text = tokenizer(instruction, padding='max_length', truncation=True,
                     max_length=config.tokenizer_max_length, return_tensors='pt')
    rng = np.random.default_rng(20261029)
    images = []
    tensors = {}
    for i in range(3):
        raw = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        tensors[f'image.{i}.rgb'] = torch.from_numpy(raw.copy())
        image = torch.from_numpy(raw).permute(2, 0, 1).unsqueeze(0).float() / 255
        image = resize_with_pad(image, 512, 512, pad_value=0) * 2 - 1
        images.append(image)
        tensors[f'image.{i}.prepared'] = image
    state = torch.zeros((1, config.max_state_dim), dtype=torch.float32)
    state[0, :6] = torch.tensor([0.125, -0.25, 0.375, -0.5, 0.625, -0.75])
    noise = torch.from_numpy(rng.standard_normal((1, config.chunk_size, config.max_action_dim)).astype(np.float32))
    tensors.update(state=state, noise=noise, language_tokens=text['input_ids'], language_mask=text['attention_mask'].bool())
    model = policy.model
    masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    with torch.inference_mode():
        for i, image in enumerate(images):
            tensors[f'image.{i}.embedding'] = model.vlm_with_expert.embed_image(image).float()
        prefix, pad, att = model.embed_prefix(images, masks, text['input_ids'], text['attention_mask'].bool(), state)
        mask2d = make_att_2d_masks(pad, att)
        positions = torch.cumsum(pad, dim=1) - 1
        tensors.update(prefix_embedding=prefix.float(), prefix_padding=pad, prefix_attention=mask2d, prefix_positions=positions)
        _, cache = model.vlm_with_expert.forward(attention_mask=mask2d, position_ids=positions,
            past_key_values=None, inputs_embeds=[prefix, None], use_cache=config.use_cache)
        for i, layer in enumerate(cache.layers):
            tensors[f'cache.{i}.key'] = layer.keys.float()
            tensors[f'cache.{i}.value'] = layer.values.float()
        captured_expert = []
        hook = model.action_out_proj.register_forward_pre_hook(
            lambda _module, inputs: captured_expert.append(inputs[0].detach().float().clone()))
        captured_mlp = {}
        captured_layer_input, captured_layer_attention, captured_layer_residual = {}, {}, {}
        captured_qkv = {}
        mlp_hooks = []
        for layer_index, layer in enumerate(model.vlm_with_expert.lm_expert.layers):
            def capture_mlp(_module, inputs, output, index=layer_index):
                captured_mlp[index] = (inputs[0].detach().float().clone(), output.detach().float().clone())
            mlp_hooks.append(layer.mlp.register_forward_hook(capture_mlp))
            def capture_input(_module, inputs, index=layer_index):
                captured_layer_input[index] = inputs[0].detach().float().clone()
            def capture_attention(_module, inputs, index=layer_index):
                captured_layer_attention[index] = inputs[0].detach().float().clone()
            def capture_residual(_module, inputs, index=layer_index):
                captured_layer_residual[index] = inputs[0].detach().clone()
            mlp_hooks.append(layer.input_layernorm.register_forward_pre_hook(capture_input))
            mlp_hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(capture_attention))
            mlp_hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(capture_residual))
            for projection in ('q_proj', 'k_proj', 'v_proj'):
                def capture_projection(_module, inputs, output, index=layer_index, name=projection):
                    captured_qkv[(index, name)] = (inputs[0].detach().float().clone(), output.detach().float().clone())
                mlp_hooks.append(getattr(layer.self_attn, projection).register_forward_hook(capture_projection))
        captured_norm_input = []
        norm_hook = model.vlm_with_expert.lm_expert.norm.register_forward_pre_hook(
            lambda _module, inputs: captured_norm_input.append(inputs[0].detach().float().clone()))
        x = noise.clone()
        dt = -1.0 / config.num_steps
        for step in range(config.num_steps):
            t = torch.tensor([1.0 + step * dt], dtype=torch.float32)
            suffix, suffix_pad, suffix_att = model.embed_suffix(x, t)
            tensors[f'step.{step}.input'] = x.clone()
            tensors[f'step.{step}.suffix'] = suffix.float()
            tensors[f'step.{step}.velocity'] = model.denoise_step(pad, cache, x, t).float()
            tensors[f'step.{step}.expert_output'] = captured_expert[-1]
            tensors[f'step.{step}.expert_norm_input'] = captured_norm_input[-1]
            for (layer_index, name), (projection_input, projection_output) in captured_qkv.items():
                tensors[f'step.{step}.layer.{layer_index}.{name}.input'] = projection_input
                tensors[f'step.{step}.layer.{layer_index}.{name}.output'] = projection_output
            for layer_index, (mlp_input, mlp_output) in captured_mlp.items():
                tensors[f'step.{step}.layer.{layer_index}.mlp_input'] = mlp_input
                tensors[f'step.{step}.layer.{layer_index}.mlp_output'] = mlp_output
                tensors[f'step.{step}.layer.{layer_index}.hidden_input'] = captured_layer_input[layer_index]
                tensors[f'step.{step}.layer.{layer_index}.attention_output'] = captured_layer_attention[layer_index]
                tensors[f'step.{step}.layer.{layer_index}.layer_output'] = (
                    captured_layer_residual[layer_index] + mlp_output.to(captured_layer_residual[layer_index].dtype)).float()
            x = x + dt * tensors[f'step.{step}.velocity']
        tensors['action.padded_normalized'] = x
        tensors['action.normalized'] = x[:, :, :config.action_feature.shape[0]]
        hook.remove()
        norm_hook.remove()
        for mlp_hook in mlp_hooks:
            mlp_hook.remove()
        # Independent reference API checks the manually captured integration loop.
        actual = model.sample_actions(images, masks, text['input_ids'], text['attention_mask'].bool(), state, noise=noise)
        if not torch.allclose(x, actual, rtol=1e-5, atol=1e-5):
            raise RuntimeError('captured Euler integration differs from sample_actions')
    tensors = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    save_file(tensors, args.output / 'reference.safetensors')
    (args.output / 'metadata.json').write_text(json.dumps({'instruction': instruction, 'state': 'already normalized',
        'actions': 'normalized, no postprocessor', 'seed': 20261029, 'threads': args.threads,
        'torch_version': torch.__version__, 'shapes': {k: list(v.shape) for k, v in tensors.items()}}, indent=2) + '\n')
    print(f'Wrote {len(tensors)} reference tensors to {args.output}')


if __name__ == '__main__':
    main()
