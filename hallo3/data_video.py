import io
import os
import sys
from functools import partial
import math
import torchvision.transforms as TT
from sgm.webds import MetaDistributedWebDataset
import random
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torchvision.io.video import av
from torchvision.utils import save_image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import decord
from decord import VideoReader
from torch.utils.data import Dataset
import json
from PIL import Image
import ast

from icecream import ic

def read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    info = {}
    audio_frames = []
    audio_timebase = _video_opt.default_timebase

    with av.open(filename, metadata_errors="ignore") as container:
        if container.streams.audio:
            audio_timebase = container.streams.audio[0].time_base
        if container.streams.video:
            video_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],
                {"video": 0},
            )
            video_fps = container.streams.video[0].average_rate
            # guard against potentially corrupted files
            if video_fps is not None:
                info["video_fps"] = float(video_fps)

        if container.streams.audio:
            audio_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.audio[0],
                {"audio": 0},
            )
            info["audio_fps"] = container.streams.audio[0].rate

    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        if pts_unit == "sec":
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            if end_pts != float("inf"):
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr

def resize_for_square_padding(arr, image_size):
    
    arr = resize(
            arr,
            size=[image_size[0], image_size[0]],
            interpolation=InterpolationMode.BICUBIC,
        )

    t, c, h, w = arr.shape

    assert h==w

    padding_width = image_size[1] - w
    
    # Calculate padding values
    pad_left = padding_width // 2
    pad_right = padding_width - pad_left
    
    # Apply padding
    arr = F.pad(arr, (pad_left, pad_right, 0, 0), mode='constant', value=0)

    return arr

def resize_only(arr, image_size):
    arr = resize(
            arr,
            size=[image_size[0], image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    return arr

def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]

def pad_last_audio(tensor, num_frames):
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]

def load_video(
    video_data,
    sampling="uniform",
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    if nb_read_frames is not None:
        ori_vlen = nb_read_frames
    else:
        ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    max_seek = int(ori_vlen - skip_frms_num - num_frames / wanted_fps * actual_fps)
    start = random.randint(skip_frms_num, max_seek + 1)
    end = int(start + num_frames / wanted_fps * actual_fps)
    n_frms = num_frames

    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / n_frms).astype(int)
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(np.arange(start, end))
    assert temp_frms is not None
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]

    return pad_last_frame(tensor_frms, num_frames)


import threading


def load_video_with_timeout(*args, **kwargs):
    video_container = {}

    def target_function():
        video = load_video(*args, **kwargs)
        video_container["video"] = video

    thread = threading.Thread(target=target_function)
    thread.start()
    timeout = 20
    thread.join(timeout)

    if thread.is_alive():
        print("Loading video timed out")
        raise TimeoutError
    return video_container.get("video", None).contiguous()


def process_video(
    video_path,
    image_size=None,
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    """
    video_path: str or io.BytesIO
    image_size: .
    duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.
    num_frames: wanted num_frames.
    wanted_fps: .
    skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
    """

    video = load_video_with_timeout(
        video_path,
        duration=duration,
        num_frames=num_frames,
        wanted_fps=wanted_fps,
        actual_fps=actual_fps,
        skip_frms_num=skip_frms_num,
        nb_read_frames=nb_read_frames,
    )

    # --- copy and modify the image process ---
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

    # resize
    if image_size is not None:
        video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")

    return video


def process_fn_video(src, image_size, fps, num_frames, skip_frms_num=0.0, txt_key="caption"):
    while True:
        r = next(src)
        if "mp4" in r:
            video_data = r["mp4"]
        elif "avi" in r:
            video_data = r["avi"]
        else:
            print("No video data found")
            continue

        if txt_key not in r:
            txt = ""
        else:
            txt = r[txt_key]

        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        else:
            txt = str(txt)

        duration = r.get("duration", None)
        if duration is not None:
            duration = float(duration)
        else:
            continue

        actual_fps = r.get("fps", None)
        if actual_fps is not None:
            actual_fps = float(actual_fps)
        else:
            continue

        required_frames = num_frames / fps * actual_fps + 2 * skip_frms_num
        required_duration = num_frames / fps + 2 * skip_frms_num / actual_fps

        if duration is not None and duration < required_duration:
            continue

        try:
            frames = process_video(
                io.BytesIO(video_data),
                num_frames=num_frames,
                wanted_fps=fps,
                image_size=image_size,
                duration=duration,
                actual_fps=actual_fps,
                skip_frms_num=skip_frms_num,
            )
            frames = (frames - 127.5) / 127.5
        except Exception as e:
            print(e)
            continue

        item = {
            "mp4": frames,
            "txt": txt,
            "num_frames": num_frames,
            "fps": fps,
        }

        yield item


class VideoDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        image_size,
        num_frames,
        fps,
        skip_frms_num=0.0,
        nshards=sys.maxsize,
        seed=1,
        meta_names=None,
        shuffle_buffer=1000,
        include_dirs=None,
        txt_key="caption",
        **kwargs,
    ):
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []

        if path.startswith(";"):
            path, include_dirs = path.split(";", 1)
        super().__init__(
            path,
            partial(
                process_fn_video, num_frames=num_frames, image_size=image_size, fps=fps, skip_frms_num=skip_frms_num
            ),
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs,
        )

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(path, **kwargs)


class SFTDataset(Dataset):
    def __init__(self, data_meta_path, video_size, fps, max_num_frames, 
                 frame_interval=1,
                 skip_frms_num=3,
                 audio_margin=2,
                 audio_type="vocals",
                 model_scale="base",
                 features="all"):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SFTDataset, self).__init__()

        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        self.audio_margin = audio_margin
        self.audio_type = audio_type
        self.audio_model = model_scale
        self.audio_features = features
        self.frame_interval = frame_interval

        vid_meta = []
        # for data_meta_path in data_meta_paths:
        with open(data_meta_path, "r", encoding="utf-8") as f:
            vid_meta.extend(json.load(f))
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)
        
        self.to_tensor = TT.ToTensor()
        self.latent_size = [60, 90]
    
    def get_mask(self, mask_bbox, video_size, h, w):
        x_min_ratio, y_min_ratio = mask_bbox[0], mask_bbox[1]
        x_max_ratio, y_max_ratio = mask_bbox[2], mask_bbox[3]
        x_min = int(x_min_ratio * w)
        y_min = int(y_min_ratio * h)
        x_max = int(x_max_ratio * w)
        y_max = int(y_max_ratio * h)
        
        mask = torch.zeros((h, w), dtype=torch.uint8)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = resize_only(mask, video_size)
        
        return mask


    def __getitem__(self, index):
        decord.bridge.set_bridge("torch")


        video_meta = self.vid_meta[index]
        
        if "bbox" in video_meta.keys():
                    
            video_path = video_meta["video_path"]

            face_emb_path = video_meta["face_emb_path"]
            caption = str(video_meta["caption"])
            
            mask_json = video_meta["bbox"]
            with open(mask_json, 'r', encoding='utf-8') as f:
                bbox = json.load(f)
            
            face_emb = torch.load(face_emb_path)
            if not isinstance(face_emb, torch.Tensor):
                face_emb = torch.tensor(face_emb)

            vr = VideoReader(uri=video_path, height=-1, width=-1)
            ori_vlen = len(vr)
            
            sample_len = self.max_num_frames * self.frame_interval

            assert ori_vlen > sample_len, video_path
            start = random.randint(
                    0, 
                    ori_vlen - sample_len - 1
                )
            
            end = min(start + self.max_num_frames * self.frame_interval, ori_vlen)
            ori_indices = np.arange(start, end, self.frame_interval).astype(int)
            temp_frms = vr.get_batch(np.arange(start, end))
            assert temp_frms is not None
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            
            ori_indices = torch.from_numpy(ori_indices)
            new_indices = torch.tensor((ori_indices - start).tolist())
            tensor_frms = tensor_frms[new_indices]

            ref_idx = random.randint(
                    0, 
                    ori_vlen-1
                )
            
            ref_image = vr[ref_idx]

            tensor_ref = torch.from_numpy(ref_image) if type(ref_image) is not torch.Tensor else ref_image
            tensor_ref = tensor_ref.permute(2, 0, 1).unsqueeze(0)
            _, _, h, w = tensor_ref.shape
            tensor_ref = resize_only(tensor_ref, self.video_size)
            
            mask_bbox = bbox[ref_idx]
            ref_mask = self.get_mask(mask_bbox, self.video_size, h, w)
            mask_ref = tensor_ref * ref_mask
            

            tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
            tensor_frms = resize_only(tensor_frms, self.video_size)
            
            assert tensor_frms.shape[0]==self.max_num_frames
            
            mask_ref = (mask_ref - 127.5) / 127.5
            tensor_frms = (tensor_frms - 127.5) / 127.5
            tensor_ref = (tensor_ref - 127.5) / 127.5
            
            item = {
                "mp4": tensor_frms,
                "txt": caption,
                "num_frames": self.max_num_frames,
                "fps": self.fps,
                "ref_image": tensor_ref,
                "face_emb": face_emb,
                "mask_ref": mask_ref
            }
            
        else:
            video_path = video_meta["video_path"]

            face_emb_path = video_meta["face_emb_path"]
            caption = str(video_meta["caption"])
            
            face_emb = torch.load(face_emb_path)
            if not isinstance(face_emb, torch.Tensor):
                face_emb = torch.tensor(face_emb)

            vr = VideoReader(uri=video_path, height=-1, width=-1)
            ori_vlen = len(vr)
            
            sample_len = self.max_num_frames * self.frame_interval

            assert ori_vlen > sample_len, video_path
            start = random.randint(
                    0, 
                    ori_vlen - sample_len - 1
                )
            
            end = min(start + self.max_num_frames * self.frame_interval, ori_vlen)
            ori_indices = np.arange(start, end, self.frame_interval).astype(int)
            temp_frms = vr.get_batch(np.arange(start, end))
            assert temp_frms is not None
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            
            ori_indices = torch.from_numpy(ori_indices)
            new_indices = torch.tensor((ori_indices - start).tolist())
            tensor_frms = tensor_frms[new_indices]

            ref_idx = random.randint(
                    0, 
                    ori_vlen-1
                )
            
            ref_image = vr[ref_idx]
            tensor_ref = torch.from_numpy(ref_image) if type(ref_image) is not torch.Tensor else ref_image
            tensor_ref = tensor_ref.permute(2, 0, 1).unsqueeze(0)
            _, _, h, w = tensor_ref.shape
            
            mask_path = video_meta["face_mask_union_path"]
            mask_image = Image.open(mask_path)
            mask = self.to_tensor(mask_image).unsqueeze(0)
            mask_ref = tensor_ref * mask
            
            tensor_ref = resize_for_square_padding(tensor_ref, self.video_size)
            mask_ref = resize_for_square_padding(mask_ref, self.video_size)
            
            tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
            tensor_frms = resize_for_square_padding(tensor_frms, self.video_size)
            
            assert tensor_frms.shape[0]==self.max_num_frames
            
            
            tensor_frms = (tensor_frms - 127.5) / 127.5
            tensor_ref = (tensor_ref - 127.5) / 127.5
            mask_ref = (mask_ref - 127.5) / 127.5
            
            item = {
                "mp4": tensor_frms,
                "txt": caption,
                "num_frames": self.max_num_frames,
                "fps": self.fps,
                "ref_image": tensor_ref,
                "face_emb": face_emb,
                "mask_ref": mask_ref
            }

        return item


        

    def __len__(self):
        return len(self.vid_meta)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        # print(path)
        return cls(data_meta_path=path, **kwargs)
    

class Stage2_SFTDataset(Dataset):
    def __init__(self, data_meta_path, video_size, fps, max_num_frames, 
                 frame_interval=1,
                 skip_frms_num=3,
                 audio_margin=2,
                 audio_type="vocals",
                 model_scale="base",
                 features="all"):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(Stage2_SFTDataset, self).__init__()

        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        self.audio_margin = audio_margin
        self.audio_type = audio_type
        self.audio_model = model_scale
        self.audio_features = features
        self.frame_interval = frame_interval

        vid_meta = []
        # for data_meta_path in data_meta_paths:
        with open(data_meta_path, "r", encoding="utf-8") as f:
            vid_meta.extend(json.load(f))
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)
        
        self.to_tensor = TT.ToTensor()
        self.latent_size = [60, 90]
    
    def get_mask(self, mask_bbox, video_size, h, w):
        x_min_ratio, y_min_ratio = mask_bbox[0], mask_bbox[1]
        x_max_ratio, y_max_ratio = mask_bbox[2], mask_bbox[3]
        x_min = int(x_min_ratio * w)
        y_min = int(y_min_ratio * h)
        x_max = int(x_max_ratio * w)
        y_max = int(y_max_ratio * h)
        
        mask = torch.zeros((h, w), dtype=torch.uint8)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = resize_only(mask, video_size)
        
        return mask


    def __getitem__(self, index):
        decord.bridge.set_bridge("torch")

        video_meta = self.vid_meta[index]

        if "bbox" in video_meta.keys():
                    
            video_path = video_meta["video_path"]

            face_emb_path = video_meta["face_emb_path"]
            caption = str(video_meta["caption"])
            
            mask_json = video_meta["bbox"]
            with open(mask_json, 'r', encoding='utf-8') as f:
                bbox = json.load(f)
            
            face_emb = torch.load(face_emb_path)
            if not isinstance(face_emb, torch.Tensor):
                face_emb = torch.tensor(face_emb)
                
            audio_emb_path = video_meta[
                f"{self.audio_type}_emb_{self.audio_model}_{self.audio_features}"
            ]
            audio_emb = torch.load(audio_emb_path)
            margin_indices = (
                torch.arange(2 * self.audio_margin + 1) - self.audio_margin
            )  # Generates [-2, -1, 0, 1, 2]
            
            vr = VideoReader(uri=video_path, height=-1, width=-1)
            ori_vlen = len(vr)
            
            sample_len = self.max_num_frames * self.frame_interval

            assert ori_vlen > sample_len+self.audio_margin, video_path
            start = random.randint(
                    self.skip_frms_num, 
                    ori_vlen - sample_len - self.audio_margin - 1
                )
            
            end = min(start + self.max_num_frames * self.frame_interval, ori_vlen)
            ori_indices = np.arange(start, end, self.frame_interval).astype(int)
            temp_frms = vr.get_batch(np.arange(start, end))
            assert temp_frms is not None
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            
            ori_indices = torch.from_numpy(ori_indices)
            new_indices = torch.tensor((ori_indices - start).tolist())
            tensor_frms = tensor_frms[new_indices]
            
            center_indices = ori_indices.unsqueeze(1) + margin_indices.unsqueeze(0)
            audio_tensor = audio_emb[center_indices]

            if random.random() < 0.05:
                audio_tensor = torch.zeros_like(audio_tensor)

            ref_idx = random.randint(
                    0, 
                    ori_vlen-1
                )
            
            ref_image = vr[ref_idx]
            tensor_ref = torch.from_numpy(ref_image) if type(ref_image) is not torch.Tensor else ref_image
            tensor_ref = tensor_ref.permute(2, 0, 1).unsqueeze(0)
            _, _, h, w = tensor_ref.shape
            tensor_ref = resize_only(tensor_ref, self.video_size)
            
            mask_bbox = bbox[ref_idx]
            ref_mask = self.get_mask(mask_bbox, self.video_size, h, w)
            mask_ref = tensor_ref * ref_mask
            
            tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
            tensor_frms = resize_only(tensor_frms, self.video_size)
            
            assert tensor_frms.shape[0]==self.max_num_frames
            assert tensor_frms.shape[0]==audio_tensor.shape[0], print(tensor_frms.shape[0], audio_tensor.shape[0])
                
            mask_ref = (mask_ref - 127.5) / 127.5
            tensor_frms = (tensor_frms - 127.5) / 127.5
            tensor_ref = (tensor_ref - 127.5) / 127.5
            
            item = {
                "mp4": tensor_frms,
                "txt": caption,
                "num_frames": self.max_num_frames,
                "fps": self.fps,
                "ref_image": tensor_ref,
                "face_emb": face_emb,
                "mask_ref": mask_ref,
                "audio_emb": audio_tensor
            }
            
        else:
            video_path = video_meta["video_path"]

            face_emb_path = video_meta["face_emb_path"]
            caption = str(video_meta["caption"])
            
            face_emb = torch.load(face_emb_path)
            if not isinstance(face_emb, torch.Tensor):
                face_emb = torch.tensor(face_emb)
            
                
            audio_emb_path = video_meta[
                f"{self.audio_type}_emb_{self.audio_model}_{self.audio_features}"
            ]
            audio_emb = torch.load(audio_emb_path)
            margin_indices = (
                torch.arange(2 * self.audio_margin + 1) - self.audio_margin
            )  # Generates [-2, -1, 0, 1, 2]

            vr = VideoReader(uri=video_path, height=-1, width=-1)
            ori_vlen = len(vr)
            
            sample_len = self.max_num_frames * self.frame_interval

            assert ori_vlen > sample_len+self.audio_margin, video_path
            start = random.randint(
                    self.skip_frms_num, 
                    ori_vlen - sample_len - self.audio_margin - 1
                )
            
            end = min(start + self.max_num_frames * self.frame_interval, ori_vlen)
            ori_indices = np.arange(start, end, self.frame_interval).astype(int)
            temp_frms = vr.get_batch(np.arange(start, end))
            assert temp_frms is not None
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            
            ori_indices = torch.from_numpy(ori_indices)
            new_indices = torch.tensor((ori_indices - start).tolist())
            tensor_frms = tensor_frms[new_indices]
            
            center_indices = ori_indices.unsqueeze(1) + margin_indices.unsqueeze(0)
            audio_tensor = audio_emb[center_indices]

            if random.random() < 0.05:
                audio_tensor = torch.zeros_like(audio_tensor)

            ref_idx = random.randint(
                    0, 
                    ori_vlen-1
                )
            
            ref_image = vr[ref_idx]
            tensor_ref = torch.from_numpy(ref_image) if type(ref_image) is not torch.Tensor else ref_image
            tensor_ref = tensor_ref.permute(2, 0, 1).unsqueeze(0)
            _, _, h, w = tensor_ref.shape
            
            try:
                mask_path = video_meta["face_mask_union_path"]
            except:
                mask_path = video_meta["mask_path"]
            mask_image = Image.open(mask_path)
            mask = self.to_tensor(mask_image).unsqueeze(0)
            mask_ref = tensor_ref * mask
            
            tensor_ref = resize_for_square_padding(tensor_ref, self.video_size)
            mask_ref = resize_for_square_padding(mask_ref, self.video_size)
            
            tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
            tensor_frms = resize_for_square_padding(tensor_frms, self.video_size)
            
            assert tensor_frms.shape[0]==self.max_num_frames
            assert tensor_frms.shape[0]==audio_tensor.shape[0], print(tensor_frms.shape[0], audio_tensor.shape[0])
            
            tensor_frms = (tensor_frms - 127.5) / 127.5
            tensor_ref = (tensor_ref - 127.5) / 127.5
            mask_ref = (mask_ref - 127.5) / 127.5
            
            item = {
                "mp4": tensor_frms,
                "txt": caption,
                "num_frames": self.max_num_frames,
                "fps": self.fps,
                "ref_image": tensor_ref,
                "face_emb": face_emb,
                "mask_ref": mask_ref,
                "audio_emb": audio_tensor
            }

        return item

    def __len__(self):
        return len(self.vid_meta)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_meta_path=path, **kwargs)

