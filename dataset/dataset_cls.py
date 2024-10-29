import os
import json
import jsonlines
import torch
import math
import random
import cv2

from tqdm import tqdm
from collections import OrderedDict

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import subprocess
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F


class ImageTextDataset(Dataset):
    """
        Usage:
            The dataset class for image-text pairs, used for image generation training
            It supports multi-aspect ratio training
        params:
            anno_file: The annotation file list
            add_normalize: whether to normalize the input image pixel to [-1, 1], default: True
            ratios: The aspect ratios during training, format: width / height
            sizes: The resoultion of training images, format: (width, height)
    """
    def __init__(
        self, anno_file, add_normalize=True,
        ratios=[1/1, 3/5, 5/3], 
        sizes=[(1024, 1024), (768, 1280), (1280, 768)],
        crop_mode='random', p_random_ratio=0.0,
    ):  
        # Ratios and Sizes : (w h)
        super().__init__()
        
        self.image_annos = []
        if not isinstance(anno_file, list):
            anno_file = [anno_file]

        for anno_file_ in anno_file:
            print(f"Load image annotation files from {anno_file_}")
            with jsonlines.open(anno_file_, 'r') as reader:
                for item in reader:
                    self.image_annos.append(item)

        print(f"Totally Remained {len(self.image_annos)} images")

        transform_list = [
            transforms.ToTensor(),
        ]    
    
        if add_normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        
        self.transform = transforms.Compose(transform_list)

        print(f"Transform List is {transform_list}")

        assert crop_mode in ['center', 'random']
        self.crop_mode = crop_mode
        self.ratios = ratios
        self.sizes = sizes
        self.p_random_ratio = p_random_ratio

    def get_closest_size(self, x):
        if self.p_random_ratio > 0 and np.random.rand() < self.p_random_ratio:
            best_size_idx = np.random.randint(len(self.ratios))
        else:
            w, h = x.width, x.height
            best_size_idx = np.argmin([abs(w/h-r) for r in self.ratios])
        return self.sizes[best_size_idx]

    def get_resize_size(self, orig_size, tgt_size):
        if (tgt_size[1]/tgt_size[0] - 1) * (orig_size[1]/orig_size[0] - 1) >= 0:
            alt_min = int(math.ceil(max(tgt_size)*min(orig_size)/max(orig_size)))
            resize_size = max(alt_min, min(tgt_size))
        else:
            alt_max = int(math.ceil(min(tgt_size)*max(orig_size)/min(orig_size)))
            resize_size = max(alt_max, max(tgt_size))
        return resize_size

    def __len__(self):
        return len(self.image_annos)

    def __getitem__(self, index):
        image_anno = self.image_annos[index]

        try:
            img = Image.open(image_anno['image']).convert("RGB")
            text = image_anno['text']

            assert isinstance(text, str), "Text should be str"

            size = self.get_closest_size(img)
            resize_size = self.get_resize_size((img.width, img.height), size)

            img = transforms.functional.resize(img, resize_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        
            if self.crop_mode == 'center':
                img = transforms.functional.center_crop(img, (size[1], size[0]))
            elif self.crop_mode == 'random':
                img = transforms.RandomCrop((size[1], size[0]))(img)
            else:
                img = transforms.functional.center_crop(img, (size[1], size[0]))

            image_tensor = self.transform(img)
            
            return {
                "video": image_tensor,    # using keyname `video`, to be compatible with video
                "text" : text,
                "identifier": 'image',
            }
            
        except Exception as e:
            print(f'Load Image Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))


class LengthGroupedVideoTextDataset(Dataset):
    """
        Usage:
            The dataset class for video-text pairs, used for video generation training
            It groups the video with the same frames together
            Now only supporting fixed resolution during training
        params:
            anno_file: The annotation file list
            max_frames: The maximum temporal lengths (This is the vae latent temporal length) 16 => (16 - 1) * 8 + 1 = 121 frames
            load_vae_latent: Loading the pre-extracted vae latents during training, we recommend to extract the latents in advance
                to reduce the time cost per batch
            load_text_fea: Loading the pre-extracted text features during training, we recommend to extract the prompt textual features
                in advance, since the T5 encoder will cost many GPU memories
    """
    
    def __init__(self, anno_file, max_frames=16, resolution='384p', load_vae_latent=True, load_text_fea=True):
        super().__init__()

        self.video_annos = []
        self.max_frames = max_frames
        self.load_vae_latent = load_vae_latent
        self.load_text_fea = load_text_fea
        self.resolution = resolution

        assert load_vae_latent, "Now only support loading vae latents, we will support to directly load video frames in the future"

        if not isinstance(anno_file, list):
            anno_file = [anno_file]

        for anno_file_ in anno_file:
            with jsonlines.open(anno_file_, 'r') as reader:
                for item in tqdm(reader):
                    self.video_annos.append(item)
        
        print(f"Totally Remained {len(self.video_annos)} videos") 

    def __len__(self):
        return len(self.video_annos)

    def __getitem__(self, index):
        try:
            video_anno = self.video_annos[index]
            text = video_anno['text']
            latent_path = video_anno['latent']
            latent = torch.load(latent_path, map_location='cpu')  # loading the pre-extracted video latents

            # TODO: remove the hard code latent shape checking
            if self.resolution == '384p':
                assert latent.shape[-1] == 640 // 8
                assert latent.shape[-2] == 384 // 8
            else:
                assert self.resolution == '768p'
                assert latent.shape[-1] == 1280 // 8
                assert latent.shape[-2] == 768 // 8

            cur_temp = latent.shape[2]
            cur_temp = min(cur_temp, self.max_frames)

            video_latent = latent[:,:,:cur_temp].float()
            assert video_latent.shape[1] == 16

            if self.load_text_fea:
                text_fea_path = video_anno['text_fea']
                text_fea = torch.load(text_fea_path, map_location='cpu')
                return {
                    'video': video_latent,
                    'prompt_embed': text_fea['prompt_embed'],
                    'prompt_attention_mask': text_fea['prompt_attention_mask'],
                    'pooled_prompt_embed': text_fea['pooled_prompt_embed'],
                    "identifier": 'video',
                }

            else:
                return {
                    'video': video_latent,
                    'text': text,
                    "identifier": 'video',
                }

        except Exception as e:
            print(f'Load Video Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))


class VideoFrameProcessor:
    # load a video and transform
    def __init__(self, resolution=256, num_frames=24, add_normalize=True, sample_fps=24):
    
        image_size = resolution

        transform_list = [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
        ]
        
        if add_normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        print(f"Transform List is {transform_list}")
        self.num_frames = num_frames
        self.transform = transforms.Compose(transform_list)
        self.sample_fps = sample_fps

    def __call__(self, video_path):
        try:
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frames = []

            while True:
                flag, frame = video_capture.read()
                if not flag:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)

            video_capture.release()
            sample_fps = self.sample_fps
            interval = max(int(fps / sample_fps), 1)
            frames = frames[::interval]

            if len(frames) < self.num_frames:
                num_frame_to_pack = self.num_frames - len(frames)
                recurrent_num = num_frame_to_pack // len(frames)
                frames = frames + recurrent_num * frames + frames[:(num_frame_to_pack % len(frames))]
                assert len(frames) >= self.num_frames, f'{len(frames)}'

            start_indexs = list(range(0, max(0, len(frames) - self.num_frames + 1)))            
            start_index = random.choice(start_indexs)

            filtered_frames = frames[start_index : start_index+self.num_frames]
            assert len(filtered_frames) == self.num_frames, f"The sampled frames should equals to {self.num_frames}"

            filtered_frames = torch.stack(filtered_frames).float() / 255
            filtered_frames = self.transform(filtered_frames)
            filtered_frames = filtered_frames.permute(1, 0, 2, 3)

            return filtered_frames, None
            
        except Exception as e:
            print(f"Load video: {video_path} Error, Exception {e}")
            return None, None


class VideoDataset(Dataset):
    def __init__(self, anno_file, resolution=256, max_frames=6, add_normalize=True):
        super().__init__()

        self.video_annos = []
        self.max_frames = max_frames

        if not isinstance(anno_file, list):
            anno_file = [anno_file]

        print(f"The training video clip frame number is {max_frames} ")

        for anno_file_ in anno_file:
            print(f"Load annotation file from {anno_file_}")

            with jsonlines.open(anno_file_, 'r') as reader:
                for item in tqdm(reader):
                    self.video_annos.append(item)
        
        print(f"Totally Remained {len(self.video_annos)} videos")
        
        self.video_processor = VideoFrameProcessor(resolution, max_frames, add_normalize)        

    def __len__(self):
        return len(self.video_annos)

    def __getitem__(self, index):
        video_anno = self.video_annos[index]
        video_path = video_anno['video']

        try:
            video_tensors, video_frames = self.video_processor(video_path)

            assert video_tensors.shape[1] == self.max_frames
            
            return {
                "video": video_tensors,
                "identifier": 'video',
            }

        except Exception as e:
            print('Loading Video Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))


class ImageDataset(Dataset):
    def __init__(self, anno_file, resolution=256, max_frames=8, add_normalize=True):
        super().__init__()

        self.image_annos = []
        self.max_frames = max_frames   
        image_paths = []

        if not isinstance(anno_file, list):
            anno_file = [anno_file]

        for anno_file_ in anno_file:
            print(f"Load annotation file from {anno_file_}")
            with jsonlines.open(anno_file_, 'r') as reader:
                for item in tqdm(reader):
                    image_paths.append(item['image'])
        
        print(f"Totally Remained {len(image_paths)} images")

        # pack multiple frames
        for idx in range(0, len(image_paths), self.max_frames):
            image_path_shard = image_paths[idx : idx + self.max_frames]
            if len(image_path_shard) < self.max_frames:
                image_path_shard = image_path_shard + image_paths[:self.max_frames - len(image_path_shard)]
            assert len(image_path_shard) == self.max_frames
            self.image_annos.append(image_path_shard)

        image_size = resolution
        transform_list = [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]    
        if add_normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        print(f"Transform List is {transform_list}")
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_annos)

    def __getitem__(self, index):
        image_paths = self.image_annos[index]

        try:
            packed_pil_frames = [Image.open(image_path).convert("RGB") for image_path in image_paths]
            filtered_frames = [self.transform(frame) for frame in packed_pil_frames]
            filtered_frames = torch.stack(filtered_frames)  # [t, c, h, w]
            filtered_frames = filtered_frames.permute(1, 0, 2, 3) # [c, t, h, w]
            
            return {
                "video": filtered_frames,
                "identifier": 'image',
            }

        except Exception as e:
            print(f'Load Images Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))