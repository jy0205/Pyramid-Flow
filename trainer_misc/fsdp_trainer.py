import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import accelerate
from .utils import MetricLogger, SmoothedValue


def update_ema_for_dit(model, model_ema, accelerator, decay):
    """Apply exponential moving average update.

    The  weights are updated in-place as follow:
    w_ema = w_ema * decay + (1 - decay) * w
    Args:
        model: active model that is being optimized
        model_ema: running average model
        decay: exponential decay parameter
    """
    with torch.no_grad():
        msd = accelerator.get_state_dict(model)
        for k, ema_v in model_ema.state_dict().items():
            if k in msd:
                model_v = msd[k].detach().to(ema_v.device, dtype=ema_v.dtype)
                ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)


def get_decay(optimization_step: int, ema_decay: float) -> float:
    """
    Compute the decay factor for the exponential moving average.
    """
    step = max(0, optimization_step - 1)

    if step <= 0:
        return 0.0

    cur_decay_value = (1 + step) / (10 + step)
    cur_decay_value = min(cur_decay_value, ema_decay)
    cur_decay_value = max(cur_decay_value, 0.0)

    return cur_decay_value


def train_one_epoch_with_fsdp(
    runner,
    model_ema: torch.nn.Module,
    accelerator: accelerate.Accelerator,
    model_dtype: str,
    data_loader: Iterable, 
    optimizer: torch.optim.Optimizer,
    lr_schedule_values,
    device: torch.device, 
    epoch: int, 
    clip_grad: float = 1.0,
    start_steps=None,
    args=None,
    print_freq=20,
    iters_per_epoch=2000,
    ema_decay=0.9999,
    use_temporal_pyramid=True,
):
    runner.dit.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0.0

    print("Start training epoch {}, {} iters per inner epoch. Training dtype {}".format(epoch, iters_per_epoch, model_dtype))

    for step in metric_logger.log_every(range(iters_per_epoch), print_freq, header):
        if step >= iters_per_epoch:
            break

        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule_values[start_steps] * param_group.get("lr_scale", 1.0)

        for _ in range(args.gradient_accumulation_steps):

            with accelerator.accumulate(runner.dit):
                # To fetch the data sample and Move the input to device
                samples = next(data_loader)
                video =  samples['video'].to(accelerator.device)
                text = samples['text']
                identifier = samples['identifier']

                # Perform the forward using the accerlate
                loss, log_loss = runner(video, text, identifier, 
                    use_temporal_pyramid=use_temporal_pyramid, accelerator=accelerator)

                # Check if the loss is nan
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value), force=True)
                    sys.exit(1)

                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()

                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                # clip the gradient
                if accelerator.sync_gradients:
                    params_to_clip = runner.dit.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, clip_grad)
                
                # To deal with the abnormal data point
                if train_loss >= 2.0:
                    print(f"The ERROR data sample, finding extreme high loss {train_loss}, skip updating the parameters", force=True)
                    # zero out the gradient, do not update
                    optimizer.zero_grad()
                    train_loss = 0.001    # fix the loss for logging
                else:
                    optimizer.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                # Update every 100 steps
                if model_ema is not None and start_steps % 100 == 0:
                    # cur_ema_decay = get_decay(start_steps, ema_decay)
                    cur_ema_decay = ema_decay
                    update_ema_for_dit(runner.dit, model_ema, accelerator, decay=cur_ema_decay)

                start_steps += 1

                # Report to tensorboard
                accelerator.log({"train_loss": train_loss}, step=start_steps)
                metric_logger.update(loss=train_loss)

                train_loss = 0.0

                min_lr = 10.
                max_lr = 0.
                for group in optimizer.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])

                metric_logger.update(lr=max_lr)
                metric_logger.update(min_lr=min_lr)
                weight_decay_value = None
                for group in optimizer.param_groups:
                    if group["weight_decay"] > 0:
                        weight_decay_value = group["weight_decay"]
                metric_logger.update(weight_decay=weight_decay_value)
                metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}