import os
import json
import torch
import time
import random
from typing import Iterable

from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, IterableDataset, DistributedSampler, RandomSampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F
from .bucket_loader import Bucketeer, TemporalLengthBucketeer


class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False, epoch: int = 0):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)


def identity(x):
    return x


def create_image_text_dataloaders(dataset, batch_size, num_workers, 
    multi_aspect_ratio=True, epoch=0, sizes=[(512, 512), (384, 640), (640, 384)],
    use_distributed=True, world_size=None, rank=None,
):
    """
        The dataset has already been splited by different rank
    """
    if use_distributed:
        assert world_size is not None
        assert rank is not None
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            seed=epoch,
        )
    else:
        sampler = RandomSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=identity if multi_aspect_ratio else default_collate,
        drop_last=True,
    )

    if multi_aspect_ratio:
        dataloader_iterator = Bucketeer(
            dataloader,
            sizes=sizes,
            is_infinite=True, epoch=epoch,
        )
    else:
        dataloader_iterator = iter(dataloader)

    # To make it infinite
    loader = IterLoader(dataloader_iterator, use_distributed=False, epoch=epoch)

    return loader


def create_length_grouped_video_text_dataloader(dataset, batch_size, num_workers, max_frames, 
    world_size=None, rank=None, epoch=0, use_distributed=False):
    if use_distributed:
        assert world_size is not None
        assert rank is not None
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            seed=epoch,
        )
    else:
        sampler = RandomSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=identity,
        drop_last=True,
    )

    # make it infinite
    dataloader_iterator = TemporalLengthBucketeer(
        dataloader,
        max_frames=max_frames,
        epoch=epoch,
    )

    return dataloader_iterator


def create_mixed_dataloaders(
    dataset, batch_size, num_workers, world_size=None, rank=None, epoch=0, 
    image_mix_ratio=0.1, use_image_video_mixed_training=True,
):
    """
        The video & image mixed training dataloader builder
    """

    assert world_size is not None
    assert rank is not None

    image_gpus = max(1, int(world_size * image_mix_ratio))
    if use_image_video_mixed_training:
        video_gpus = world_size - image_gpus
    else:
        # only use video data
        video_gpus = world_size
        image_gpus = 0

    print(f"{image_gpus} gpus for image, {video_gpus} gpus for video")

    if rank < video_gpus:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=video_gpus,
            rank=rank,
            seed=epoch,
        )
    else:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=image_gpus,
            rank=rank - video_gpus,
            seed=epoch,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=default_collate,
        drop_last=True,
    )

    # To make it infinite
    loader = IterLoader(loader, use_distributed=True, epoch=epoch)
    return loader