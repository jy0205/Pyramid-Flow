import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class DDPMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.Tensor


class DDPMCosineScheduler(SchedulerMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        scaler: float = 1.0,
        s: float = 0.008,
    ):
        self.scaler = scaler
        self.s = torch.tensor([s])
        self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

    def _alpha_cumprod(self, t, device):
        if self.scaler > 1:
            t = 1 - (1 - t) ** self.scaler
        elif self.scaler < 1:
            t = t**self.scaler
        alpha_cumprod = torch.cos(
            (t + self.s.to(device)) / (1 + self.s.to(device)) * torch.pi * 0.5
        ) ** 2 / self._init_alpha_cumprod.to(device)
        return alpha_cumprod.clamp(0.0001, 0.9999)

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.Tensor`: scaled input sample
        """
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        timesteps: Optional[List[int]] = None,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`Dict[float, int]`):
                the number of diffusion steps used when generating samples with a pre-trained model. If passed, then
                `timesteps` must be `None`.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps are moved to. {2 / 3: 20, 0.0: 10}
        """
        if timesteps is None:
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.Tensor(timesteps).to(device)
        self.timesteps = timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        dtype = model_output.dtype
        device = model_output.device
        t = timestep

        prev_t = self.previous_timestep(t)

        alpha_cumprod = self._alpha_cumprod(t, device).view(t.size(0), *[1 for _ in sample.shape[1:]])
        alpha_cumprod_prev = self._alpha_cumprod(prev_t, device).view(prev_t.size(0), *[1 for _ in sample.shape[1:]])
        alpha = alpha_cumprod / alpha_cumprod_prev

        mu = (1.0 / alpha).sqrt() * (sample - (1 - alpha) * model_output / (1 - alpha_cumprod).sqrt())

        std_noise = randn_tensor(mu.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
        std = ((1 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)).sqrt() * std_noise
        pred = mu + std * (prev_t != 0).float().view(prev_t.size(0), *[1 for _ in sample.shape[1:]])

        if not return_dict:
            return (pred.to(dtype),)

        return DDPMSchedulerOutput(prev_sample=pred.to(dtype))

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        device = original_samples.device
        dtype = original_samples.dtype
        alpha_cumprod = self._alpha_cumprod(timesteps, device=device).view(
            timesteps.size(0), *[1 for _ in original_samples.shape[1:]]
        )
        noisy_samples = alpha_cumprod.sqrt() * original_samples + (1 - alpha_cumprod).sqrt() * noise
        return noisy_samples.to(dtype=dtype)

    def __len__(self):
        return self.config.num_train_timesteps

    def previous_timestep(self, timestep):
        index = (self.timesteps - timestep[0]).abs().argmin().item()
        prev_t = self.timesteps[index + 1][None].expand(timestep.shape[0])
        return prev_t
