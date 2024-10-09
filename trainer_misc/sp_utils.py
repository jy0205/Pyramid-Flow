import os
import torch
import torch.distributed as dist
from .utils import is_dist_avail_and_initialized, get_rank


SEQ_PARALLEL_GROUP = None
SEQ_PARALLEL_SIZE = None
SEQ_PARALLEL_PROC_NUM = None    # using how many process for sequence parallel

SYNC_INPUT_GROUP = None
SYNC_INPUT_SIZE = None

def is_sequence_parallel_initialized():
    if SEQ_PARALLEL_GROUP is None:
        return False
    else:
        return True


def init_sequence_parallel_group(args):
    global SEQ_PARALLEL_GROUP
    global SEQ_PARALLEL_SIZE
    global SEQ_PARALLEL_PROC_NUM

    assert SEQ_PARALLEL_GROUP is None, "sequence parallel group is already initialized"
    assert is_dist_avail_and_initialized(), "The pytorch distributed should be initialized"
    SEQ_PARALLEL_SIZE = args.sp_group_size
    
    print(f"Setting the Sequence Parallel Size {SEQ_PARALLEL_SIZE}")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if args.sp_proc_num == -1:
        SEQ_PARALLEL_PROC_NUM = world_size
    else:
        SEQ_PARALLEL_PROC_NUM = args.sp_proc_num

    assert SEQ_PARALLEL_PROC_NUM % SEQ_PARALLEL_SIZE == 0, "The process needs to be evenly divided"

    for i in range(0, SEQ_PARALLEL_PROC_NUM, SEQ_PARALLEL_SIZE):
        ranks = list(range(i, i + SEQ_PARALLEL_SIZE))
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            SEQ_PARALLEL_GROUP = group
            break


def init_sync_input_group(args):
    global SYNC_INPUT_GROUP
    global SYNC_INPUT_SIZE

    assert SYNC_INPUT_GROUP is None, "parallel group is already initialized"
    assert is_dist_avail_and_initialized(), "The pytorch distributed should be initialized"
    SYNC_INPUT_SIZE = args.max_frames

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    for i in range(0, world_size, SYNC_INPUT_SIZE):
        ranks = list(range(i, i + SYNC_INPUT_SIZE))
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            SYNC_INPUT_GROUP = group
            break


def get_sequence_parallel_group():
    assert SEQ_PARALLEL_GROUP is not None, "sequence parallel group is not initialized"
    return SEQ_PARALLEL_GROUP


def get_sync_input_group():
    return SYNC_INPUT_GROUP


def get_sequence_parallel_world_size():
    assert SEQ_PARALLEL_SIZE is not None, "sequence parallel size is not initialized"
    return SEQ_PARALLEL_SIZE


def get_sequence_parallel_rank():
    assert SEQ_PARALLEL_SIZE is not None, "sequence parallel size is not initialized"
    rank = get_rank()
    cp_rank = rank % SEQ_PARALLEL_SIZE
    return cp_rank


def get_sequence_parallel_group_rank():
    assert SEQ_PARALLEL_SIZE is not None, "sequence parallel size is not initialized"
    rank = get_rank()
    cp_group_rank = rank // SEQ_PARALLEL_SIZE
    return cp_group_rank


def get_sequence_parallel_proc_num():
    return SEQ_PARALLEL_PROC_NUM
