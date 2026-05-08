from __future__ import annotations

import torch


def sample_flow_timesteps(scheduler, batch_size: int, device: torch.device, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(
        low=0,
        high=scheduler.config.num_train_timesteps,
        size=(batch_size,),
        device=device,
        generator=generator,
    ).long()
    timesteps = scheduler.timesteps.to(device=device)[indices]
    return indices, timesteps


def get_sigmas(scheduler, timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def flow_matching_loss(noise_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

