from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import math
import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class PyramidFlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,     # Following Stable diffusion 3, 
        stages: int = 3,
        stage_range: List = [0, 1/3, 2/3, 1],
        gamma: float = 1/3,
    ):
        
        self.timestep_ratios = {}           # The timestep ratio for each stage
        self.timesteps_per_stage = {}       # The  detailed timesteps per stage
        self.sigmas_per_stage = {}
        self.start_sigmas = {}           
        self.end_sigmas = {}
        self.ori_start_sigmas = {}

        # self.init_sigmas()
        self.init_sigmas_for_each_stage()
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self.gamma = gamma

    def init_sigmas(self):
        """
            initialize the global timesteps and sigmas
        """
        num_train_timesteps = self.config.num_train_timesteps
        shift = self.config.shift

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def init_sigmas_for_each_stage(self):
        """
            Init the timesteps for each stage
        """
        self.init_sigmas()

        stage_distance = []
        stages = self.config.stages
        training_steps = self.config.num_train_timesteps
        stage_range = self.config.stage_range

        # Init the start and end point of each stage
        for i_s in range(stages):
            # To decide the start and ends point
            start_indice = int(stage_range[i_s] * training_steps)
            start_indice = max(start_indice, 0)
            end_indice = int(stage_range[i_s+1] * training_steps)
            end_indice = min(end_indice, training_steps)
            start_sigma = self.sigmas[start_indice].item()
            end_sigma = self.sigmas[end_indice].item() if end_indice < training_steps else 0.0
            self.ori_start_sigmas[i_s] = start_sigma

            if i_s != 0:
                ori_sigma = 1 - start_sigma
                gamma = self.config.gamma
                corrected_sigma = (1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)) * ori_sigma
                # corrected_sigma = 1 / (2 - ori_sigma) * ori_sigma
                start_sigma = 1 - corrected_sigma

            stage_distance.append(start_sigma - end_sigma)
            self.start_sigmas[i_s] = start_sigma
            self.end_sigmas[i_s] = end_sigma

        # Determine the ratio of each stage according to flow length
        tot_distance = sum(stage_distance)
        for i_s in range(stages):
            if i_s == 0:
                start_ratio = 0.0
            else:
                start_ratio = sum(stage_distance[:i_s]) / tot_distance
            if i_s == stages - 1:
                end_ratio = 1.0
            else:
                end_ratio = sum(stage_distance[:i_s+1]) / tot_distance

            self.timestep_ratios[i_s] = (start_ratio, end_ratio)

        # Determine the timesteps and sigmas for each stage
        for i_s in range(stages):
            timestep_ratio = self.timestep_ratios[i_s]
            timestep_max = self.timesteps[int(timestep_ratio[0] * training_steps)]
            timestep_min = self.timesteps[min(int(timestep_ratio[1] * training_steps), training_steps - 1)]
            timesteps = np.linspace(
                timestep_max, timestep_min, training_steps + 1,
            )
            self.timesteps_per_stage[i_s] = timesteps[:-1] if isinstance(timesteps, torch.Tensor) else torch.from_numpy(timesteps[:-1])
            stage_sigmas = np.linspace(
                1, 0, training_steps + 1,
            )
            self.sigmas_per_stage[i_s] = torch.from_numpy(stage_sigmas[:-1])

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, stage_index: int, device: Union[str, torch.device] = None):
        """
            Setting the timesteps and sigmas for each stage 
        """
        self.num_inference_steps = num_inference_steps
        training_steps = self.config.num_train_timesteps     
        self.init_sigmas()

        stage_timesteps = self.timesteps_per_stage[stage_index]
        timestep_max = stage_timesteps[0].item()
        timestep_min = stage_timesteps[-1].item()

        timesteps = np.linspace(
            timestep_max, timestep_min, num_inference_steps,
        )
        self.timesteps = torch.from_numpy(timesteps).to(device=device)

        stage_sigmas = self.sigmas_per_stage[stage_index]
        sigma_max = stage_sigmas[0].item()
        sigma_min = stage_sigmas[-1].item()

        ratios = np.linspace(
            sigma_max, sigma_min, num_inference_steps
        )
        sigmas = torch.from_numpy(ratios).to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._step_index = 0

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
