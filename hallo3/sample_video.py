import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio

import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as TT


from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from PIL import Image
import moviepy.editor as mp
import torchvision.transforms as transforms
import torch.nn.functional as F

from sgm.utils.audio_processor import AudioProcessor
from sgm.utils.image_processor import ImageProcessor
from icecream import ic
from torchvision.utils import save_image

def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

def save_video_as_grid_and_mp4_with_audio(video_batch: torch.Tensor, save_path: str, audio_path: str, fps: int = 5, is_padding: bool = False):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            
            if is_padding:
                h, w, _ = frame.shape
                crop_size = min(w, h)  
                start_x = (w - crop_size) // 2
                cropped_frame = frame[:, start_x:start_x + crop_size]
                gif_frames.append(cropped_frame)
            else:
                gif_frames.append(frame)

        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

        video_clip = mp.VideoFileClip(now_save_path)
        audio_clip = mp.AudioFileClip(audio_path)

        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)

        video_with_audio = video_clip.set_audio(audio_clip)
 
        final_save_path = os.path.join(save_path, f"{i:06d}_with_audio.mp4")
        video_with_audio.write_videofile(final_save_path, fps=fps)
        
        os.remove(now_save_path)

        video_clip.close()
        audio_clip.close()

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

def process_audio_emb(audio_emb):
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb

def resize_for_square_padding(arr, image_size):
    arr = transforms.Resize(size=[image_size[0], image_size[0]])(arr)
    
    t, c, h, w = arr.shape

    assert h == w, "Height and width must be equal after resizing."

    padding_width = image_size[1] - w
    
    pad_left = padding_width // 2
    pad_right = padding_width - pad_left
    
    arr = F.pad(arr, (pad_left, pad_right, 0, 0), mode='constant', value=0)

    return arr

def add_mask_to_first_frame(image, mask_rate=0.25):
    b, c, f, h, w = image.shape
    image = image.permute(0, 2, 1, 3, 4).contiguous()
    rand_mask = torch.rand(h, w).to(dtype=image.dtype, device=image.device)
    mask = rand_mask > mask_rate
    mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
    mask = mask.expand(b, f, c, h, w)  

    image = image * mask
    
    image = image.permute(0, 2, 1, 3, 4).contiguous()
    return image

def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    step = None
    load_checkpoint(model, args, specific_iteration=step)
    model.eval()

    if args.input_type == "cli":
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("*********************rank and world_size", rank, world_size)
        print(args.input_file)
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        raise NotImplementedError

    image_size = [480, 720]

    if args.image2video:
        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        transform = TT.Compose(chained_trainsforms)

    audio_separator_model_file = args.audio_separator_model_path
    wav2vec_model_path = args.wav2vec_model_path
    wav2vec_only_last_features = args.wav2vec_features == "last"

    audio_processor = AudioProcessor(
                    args.sample_rate,
                    wav2vec_model_path,
                    wav2vec_only_last_features,
                    os.path.dirname(audio_separator_model_file),
                    os.path.basename(audio_separator_model_file),
                    os.path.join(".cache", "audio_preprocess")
                )
    
    image_processor = ImageProcessor(args.face_analysis_model_path)

    sample_func = model.sample
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    L = (T-1)*4 + 1


    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    device = model.device
    ic(device)
    model = model.to("cuda")
    n_motion_frame = 2
    mask_rate = 0.1
    with torch.no_grad():
        for text, cnt in tqdm(data_iter):
            assert args.image2video
            
            input_list = text.split("@@")
            assert len(input_list)==3
            text, image_path, audio_path = input_list[0], input_list[1], input_list[2]
            assert os.path.exists(image_path), image_path
            assert os.path.exists(audio_path), audio_path
            

            name = os.path.splitext(os.path.basename(image_path))[0] + "-" + os.path.splitext(os.path.basename(audio_path))[0] + f"-seed_{args.seed}"
            save_path = os.path.join(args.output_dir, name)
            os.makedirs(save_path, exist_ok=True)

            audio_emb, length = audio_processor.preprocess(audio_path, L)
            audio_emb = process_audio_emb(audio_emb) 

            face_emb, face_mask_path = image_processor.preprocess(image_path, save_path, 1.2)
            face_emb = face_emb.reshape(1, -1)
            face_emb = torch.tensor(face_emb).to("cuda")
            
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to("cuda")
            face_mask = Image.open(face_mask_path).convert("RGB")
            face_mask = transform(face_mask).unsqueeze(0).to("cuda")
            ref_image = image * face_mask
            
            _, _, h, w = image.shape
            if h==w:
                is_padding = True
            else:
                is_padding = False
            
            if is_padding:
                image = resize_for_square_padding(image, image_size).clamp(0, 1)
            else:
                image = resize_for_rectangle_crop(image, image_size, reshape_mode="center").unsqueeze(0)
            
            image = image * 2.0 - 1.0
            motion_image = image.unsqueeze(2).to(torch.bfloat16)
            ref_image_pixel = image.unsqueeze(2).to(torch.bfloat16)
            
            
            if is_padding:
                ref_image = resize_for_square_padding(ref_image, image_size).clamp(0, 1)
            else:
                ref_image = resize_for_rectangle_crop(ref_image, image_size, reshape_mode="center").unsqueeze(0)
            
            
            ref_image = ref_image * 2.0 - 1.0
            ref_image = ref_image.unsqueeze(2).to(torch.bfloat16)
            
            motion_image = torch.cat([motion_image]*n_motion_frame, dim=2)
            mask_image = add_mask_to_first_frame(motion_image, mask_rate=mask_rate)
            mask_image = torch.cat([ref_image_pixel, mask_image], dim=2)
            mask_image = model.encode_first_stage(mask_image, None)
            mask_image = mask_image.permute(0, 2, 1, 3, 4).contiguous()
            
            ref_image = model.encode_first_stage(ref_image, None)
            ref_image = ref_image.permute(0, 2, 1, 3, 4).contiguous()
                    
            pad_shape = (mask_image.shape[0], T - 1, C, H // F, W // F)
            mask_image = torch.concat([mask_image, torch.zeros(pad_shape).to(mask_image.device).to(mask_image.dtype)], dim=1)

            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            times = audio_emb.shape[0] // (L-n_motion_frame)
            if times * (L-n_motion_frame) < audio_emb.shape[0]:
                times += 1
            video = []
            pre_fix = torch.zeros_like(audio_emb[:n_motion_frame])

            # times = 1
            for t in range(times):
                print(f"[{t+1}/{times}]")

                if args.image2video and mask_image is not None:
                    c["concat"] = mask_image
                    uc["concat"] = mask_image

                assert args.batch_size == 1
                audio_tensor = audio_emb[
                    t * (L-n_motion_frame): min((t + 1) * (L-n_motion_frame), audio_emb.shape[0])
                ]
                
                audio_tensor = torch.cat([pre_fix, audio_tensor], dim=0)
                pre_fix = audio_tensor[-n_motion_frame:]
                
                if audio_tensor.shape[0]!=L:
                    pad = L - audio_tensor.shape[0]
                    assert pad > 0
                    padding = pre_fix[-1:].repeat(pad, *([1] * (pre_fix.dim() - 1)))
                    audio_tensor = torch.cat([audio_tensor, padding], dim=0)
                
                audio_tensor = audio_tensor.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                
                samples_z = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                    audio_emb=audio_tensor,
                    ref_image=ref_image,
                    face_emb=face_emb
                )

                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

                torch.cuda.empty_cache()
                latent = 1.0 / model.scale_factor * samples_z

                # Decode latent serial to save GPU memory
                recons = []
                loop_num = (T - 1) // 2
                for i in range(loop_num):
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3
                    if i == loop_num - 1:
                        clear_fake_cp_cache = True
                    else:
                        clear_fake_cp_cache = False
                    with torch.no_grad():
                        recon = model.first_stage_model.decode(
                            latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                        )

                    recons.append(recon)

                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                
                motion_image = samples[:,-n_motion_frame:].permute(0, 2, 1, 3, 4).contiguous().to(dtype=torch.bfloat16, device="cuda")
                motion_image = motion_image * 2 - 1
                mask_image = add_mask_to_first_frame(motion_image, mask_rate=mask_rate)
                mask_image = torch.cat([ref_image_pixel, mask_image], dim=2)
                mask_image = model.encode_first_stage(mask_image, None)
                mask_image = mask_image.permute(0, 2, 1, 3, 4).contiguous()
                
                mask_image = torch.concat([mask_image, torch.zeros(pad_shape).to(mask_image.device).to(mask_image.dtype)], dim=1)
                
                video.append(samples[:, n_motion_frame:])

            video = torch.cat(video, dim=1)
            video = video[:, :length]
            
            if mpu.get_model_parallel_rank() == 0:
                save_video_as_grid_and_mp4_with_audio(video, save_path, audio_path, fps=args.sampling_fps, is_padding=is_padding)
                print("saving in: ", save_path)


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    sampling_main(args, model_cls=SATVideoDiffusionEngine)
