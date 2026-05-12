import torch


def load_model_dict_into_meta(model, state_dict, device, dtype=None, model_name_or_path=None):
    """Compatibility replacement for diffusers' removed load_model_dict_into_meta."""
    from accelerate.utils import set_module_tensor_to_device

    expected_keys = set(model.state_dict().keys())
    unexpected_keys = []

    for key, value in state_dict.items():
        if key not in expected_keys:
            unexpected_keys.append(key)
            continue

        target_dtype = dtype if dtype is not None and torch.is_floating_point(value) else None
        try:
            set_module_tensor_to_device(
                model,
                key,
                device,
                value=value,
                dtype=target_dtype,
            )
        except ValueError as exc:
            source = f" from {model_name_or_path}" if model_name_or_path is not None else ""
            raise ValueError(f"Cannot load parameter {key}{source}: {exc}") from exc

    return unexpected_keys
