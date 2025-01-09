import os
import math
import gradio as gr
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
import tempfile
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT
import moviepy.editor as mp
import imageio

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args

import uuid

def save_video_as_grid_and_mp4_with_audio(video_batch: torch.Tensor, save_path: str, audio_path: str, fps: int = 5, is_padding: bool = False):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = frame.permute(1, 2, 0)  # c h w -> h w c
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

        filename_list = f"{str(uuid.uuid4())}_with_audio.mp4"
        final_save_path = os.path.join(save_path, filename_list)
        video_with_audio.write_videofile(final_save_path, fps=fps)
        
        os.remove(now_save_path)
        video_clip.close()
        audio_clip.close()
        return filename_list
        
from sgm.utils.audio_processor import AudioProcessor
from sgm.utils.image_processor import ImageProcessor

# Reuse all the helper functions from sample_video.py
def process_audio_emb(audio_emb):
    concatenated_tensors = []
    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)] for j in range(-2, 3)
        ]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))
    return torch.stack(concatenated_tensors, dim=0)

def resize_for_square_padding(arr, image_size):
    arr = transforms.Resize(size=[image_size[0], image_size[0]])(arr)
    t, c, h, w = arr.shape
    assert h == w, "Height and width must be equal after resizing."
    padding_width = image_size[1] - w
    pad_left = padding_width // 2
    pad_right = padding_width - pad_left
    arr = F.pad(arr, (pad_left, pad_right, 0, 0), mode='constant', value=0)
    return arr

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
    if reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
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

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N, T=None, device="cuda"):
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

class VideoGenerator:
    def __init__(self):
        # Initialize configuration and model
        args_list = [
            "--base", "./configs/cogvideox_5b_i2v_s2.yaml",
            "./configs/inference.yaml"
        ]
        py_parser = argparse.ArgumentParser(add_help=False)
        known, remaining_args = py_parser.parse_known_args()
        self.args = get_args(args_list)
        self.args = argparse.Namespace(**vars(self.args), **vars(known))
        
        # Configure model settings
        self.args.model_config.first_stage_config.params.cp_size = 1
        self.args.model_config.network_config.params.transformer_args.model_parallel_size = 1
        self.args.model_config.network_config.params.transformer_args.checkpoint_activations = False
        self.args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False
        
        # Initialize model
        self.model = get_model(self.args, SATVideoDiffusionEngine)
        load_checkpoint(self.model, self.args)
        self.model.eval()
        self.model = self.model.to("cuda")
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            self.args.sample_rate,
            self.args.wav2vec_model_path,
            self.args.wav2vec_features == "last",
            os.path.dirname(self.args.audio_separator_model_path),
            os.path.basename(self.args.audio_separator_model_path),
            os.path.join(".cache", "audio_preprocess")
        )
        
        self.image_processor = ImageProcessor(self.args.face_analysis_model_path)
        
        self.image_size = [480, 720]
        self.transform = TT.Compose([TT.ToTensor()])

    def generate_video(self, image, audio_file, prompt):
        with torch.no_grad():
            # Process inputs
            temp_dir = tempfile.mkdtemp()
            temp_image_path = os.path.join(temp_dir, "input_image.png")
            image.save(temp_image_path)
            
            # Process audio
            T = self.args.sampling_num_frames
            L = (T-1)*4 + 1
            audio_emb, length = self.audio_processor.preprocess(audio_file, L)
            audio_emb = process_audio_emb(audio_emb)
            
            # Process image
            face_emb, face_mask_path = self.image_processor.preprocess(temp_image_path, temp_dir, 1.2)
            face_emb = face_emb.reshape(1, -1)
            face_emb = torch.tensor(face_emb).to("cuda")
            
            # Transform image
            image_tensor = self.transform(image).unsqueeze(0).to("cuda")
            face_mask = Image.open(face_mask_path).convert("RGB")
            face_mask = self.transform(face_mask).unsqueeze(0).to("cuda")
            ref_image = image_tensor * face_mask
            
            # Check padding
            _, _, h, w = image_tensor.shape
            is_padding = h == w
            
            if is_padding:
                image_tensor = resize_for_square_padding(image_tensor, self.image_size).clamp(0, 1)
            else:
                image_tensor = resize_for_rectangle_crop(image_tensor, self.image_size, reshape_mode="center").unsqueeze(0)
            
            # Process for model input
            image_tensor = image_tensor * 2.0 - 1.0
            motion_image = image_tensor.unsqueeze(2).to(torch.bfloat16)
            ref_image_pixel = image_tensor.unsqueeze(2).to(torch.bfloat16)
            
            if is_padding:
                ref_image = resize_for_square_padding(ref_image, self.image_size).clamp(0, 1)
            else:
                ref_image = resize_for_rectangle_crop(ref_image, self.image_size, reshape_mode="center").unsqueeze(0)
            
            ref_image = ref_image * 2.0 - 1.0
            ref_image = ref_image.unsqueeze(2).to(torch.bfloat16)
            
            # Generate video
            output_path = os.path.join(temp_dir, "output_video.mp4")
            
            n_motion_frame = 2
            mask_rate = 0.1
            
            motion_image = torch.cat([motion_image]*n_motion_frame, dim=2)
            mask_image = add_mask_to_first_frame(motion_image, mask_rate=mask_rate)
            mask_image = torch.cat([ref_image_pixel, mask_image], dim=2)
            mask_image = self.model.encode_first_stage(mask_image, None)
            mask_image = mask_image.permute(0, 2, 1, 3, 4).contiguous()
            
            ref_image = self.model.encode_first_stage(ref_image, None)
            ref_image = ref_image.permute(0, 2, 1, 3, 4).contiguous()
            
            T, H, W, C, F = self.args.sampling_num_frames, self.image_size[0], self.image_size[1], self.args.latent_channels, 8
            pad_shape = (mask_image.shape[0], T - 1, C, H // F, W // F)
            mask_image = torch.concat([mask_image, torch.zeros(pad_shape).to(mask_image.device).to(mask_image.dtype)], dim=1)

            value_dict = {
                "prompt": prompt,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }

            num_samples = [1]
            force_uc_zero_embeddings = ["txt"]
            
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(self.model.conditioner),
                value_dict,
                num_samples
            )

            c, uc = self.model.conditioner.get_unconditional_conditioning(
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

            for t in range(times):
                if mask_image is not None:
                    c["concat"] = mask_image
                    uc["concat"] = mask_image

                audio_tensor = audio_emb[
                    t * (L-n_motion_frame): min((t + 1) * (L-n_motion_frame), audio_emb.shape[0])
                ]
                
                audio_tensor = torch.cat([pre_fix, audio_tensor], dim=0)
                pre_fix = audio_tensor[-n_motion_frame:]
                
                if audio_tensor.shape[0]!=L:
                    pad = L - audio_tensor.shape[0]
                    padding = pre_fix[-1:].repeat(pad, *([1] * (pre_fix.dim() - 1)))
                    audio_tensor = torch.cat([audio_tensor, padding], dim=0)
                
                audio_tensor = audio_tensor.unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
                
                samples_z = self.model.sample(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                    audio_emb=audio_tensor,
                    ref_image=ref_image,
                    face_emb=face_emb
                )

                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                latent = 1.0 / self.model.scale_factor * samples_z

                recons = []
                loop_num = (T - 1) // 2
                for i in range(loop_num):
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3
                    clear_fake_cp_cache = (i == loop_num - 1)
                    
                    with torch.no_grad():
                        recon = self.model.first_stage_model.decode(
                            latent[:, :, start_frame:end_frame].contiguous(),
                            clear_fake_cp_cache=clear_fake_cp_cache
                        )
                    recons.append(recon)

                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                
                motion_image = samples[:,-n_motion_frame:].permute(0, 2, 1, 3, 4).contiguous().to(dtype=torch.bfloat16, device="cuda")
                motion_image = motion_image * 2 - 1
                mask_image = add_mask_to_first_frame(motion_image, mask_rate=mask_rate)
                mask_image = torch.cat([ref_image_pixel, mask_image], dim=2)
                mask_image = self.model.encode_first_stage(mask_image, None)
                mask_image = mask_image.permute(0, 2, 1, 3, 4).contiguous()
                mask_image = torch.concat([mask_image, torch.zeros(pad_shape).to(mask_image.device).to(mask_image.dtype)], dim=1)
                
                video.append(samples[:, n_motion_frame:])

            video = torch.cat(video, dim=1)
            video = video[:, :length]
            
            filename_saved = save_video_as_grid_and_mp4_with_audio(video, os.path.dirname(output_path), audio_file, fps=self.args.sampling_fps, is_padding=is_padding)
            print(f"Filename: {filename_saved}")
            final_video_path = os.path.join(os.path.dirname(output_path), filename_saved)
            permanent_output_dir = "output/gradio"
            os.makedirs(permanent_output_dir, exist_ok=True)
            permanent_path = os.path.join(permanent_output_dir, f"video_{os.path.basename(final_video_path)}")
            import shutil
            shutil.copy2(final_video_path, permanent_path)
            
            print(f"Video saved to: {permanent_path}")
            return permanent_path

def create_gradio_interface():
    generator = VideoGenerator()
    
    def process(image, audio_file, prompt):
        output_video = generator.generate_video(image, audio_file, prompt)
        return output_video
    
    interface = gr.Interface(
        fn=process,
        inputs=[
            gr.Image(type="pil", label="Input Image"),
            gr.Audio(type="filepath", label="Audio File"),
            gr.Textbox(label="Prompt")
        ],
        outputs=gr.Video(label="Generated Video"),
        title="Halo 3 - Video Generation from Image and Audio",
        description="Upload an image and audio file, provide a prompt (optional), and generate a video."
    )
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)