import torch
import torch.nn as nn
import math
import torch.distributed as dist


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    concat_output: bool,
):
    if world_size == 1:
        return input_
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    if concat_output:
        return torch.cat(output_list, dim=gather_dim).contiguous()
    else:
        # For multi-gpus inference, the latent on each gpu are same, only remain the first one
        return output_list[0]


class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, process_group, world_size, scatter_dim, gather_dim, concat_output):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = world_size
        ctx.concat_output = concat_output
        output = _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim, concat_output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
            ctx.concat_output,
        )
        return (
            grad_output,
            None,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    world_size: int = 1,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    concat_output: bool = True,
):
    return _AllToAll.apply(input_, process_group, world_size, scatter_dim, gather_dim, concat_output)