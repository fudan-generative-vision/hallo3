import random

import math
from typing import Any, Dict, List, Tuple, Union
from omegaconf import ListConfig
import torch.nn.functional as F

from sat.helpers import print_rank0
import torch
from torch import nn
from transformers import AutoProcessor, CLIPVisionModel

from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)
import gc
from sat import mpu
from sgm.utils.model_io import load_checkpoint
import argparse
from copy import deepcopy

from torchvision import transforms
from PIL import Image
from icecream import ic

class SATVideoDiffusionEngine(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        model_config = args.model_config
        # model args preprocess
        log_keys = model_config.get("log_keys", None)
        input_key = model_config.get("input_key", "mp4")
        network_config = model_config.get("network_config", None)
        network_wrapper = model_config.get("network_wrapper", None)
        denoiser_config = model_config.get("denoiser_config", None)
        sampler_config = model_config.get("sampler_config", None)
        conditioner_config = model_config.get("conditioner_config", None)
        first_stage_config = model_config.get("first_stage_config", None)
        loss_fn_config = model_config.get("loss_fn_config", None)
        scale_factor = model_config.get("scale_factor", 1.0)
        latent_input = model_config.get("latent_input", False)
        disable_first_stage_autocast = model_config.get("disable_first_stage_autocast", False)
        no_cond_log = model_config.get("disable_first_stage_autocast", False)
        not_trainable_prefixes = model_config.get("not_trainable_prefixes", ["first_stage_model", "conditioner"])
        compile_model = model_config.get("compile_model", False)
        en_and_decode_n_samples_a_time = model_config.get("en_and_decode_n_samples_a_time", None)
        lr_scale = model_config.get("lr_scale", None)
        lora_train = model_config.get("lora_train", False)
        ref_network_config = model_config.get("ref_network_config", None)
        train_prefix = model_config.get("train_prefix", {})

        self.use_pd = model_config.get("use_pd", False)  # progressive distillation

        self.log_keys = log_keys
        self.input_key = input_key
        self.not_trainable_prefixes = not_trainable_prefixes
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lr_scale = lr_scale
        self.lora_train = lora_train
        self.noised_image_input = model_config.get("noised_image_input", False)
        self.noised_image_all_concat = model_config.get("noised_image_all_concat", False)
        self.noised_image_dropout = model_config.get("noised_image_dropout", 0.0)
        if args.fp16:
            dtype = torch.float16
            dtype_str = "fp16"
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = "bf16"
        else:
            dtype = torch.float32
            dtype_str = "fp32"
        self.dtype = dtype
        self.dtype_str = dtype_str
        self.train_prefix = train_prefix

        network_config["params"]["dtype"] = dtype_str
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=dtype
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None
        self.conditioner = instantiate_from_config(default(conditioner_config, UNCONDITIONAL_CONFIG))

        self._init_first_stage(first_stage_config)

        self.loss_fn = instantiate_from_config(loss_fn_config) if loss_fn_config is not None else None

        self.latent_input = latent_input
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

        ref_network_config["params"]["dtype"] = dtype_str
        ref_model = instantiate_from_config(ref_network_config)
        self.ref_model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            ref_model, compile_model=compile_model, dtype=dtype
        )
        
        if network_config.params.add_audio_module:
            self.is_s1 = False
        else:
            self.is_s1 = True
            self.ref_model_path = model_config.ref_net_path

    def disable_untrainable_params(self):
        if self.is_s1:
            args = argparse.Namespace()
            args.mode = 'finetune'
            load_checkpoint(self.ref_model, args, self.ref_model_path, prefix='model.')
        
        total_trainable = 0
        for n, p in self.named_parameters():
            if p.requires_grad == False:
                continue
            flag = True

            for name in self.train_prefix.keys():
                for prefix in self.train_prefix[name]:
                    if name in n and prefix in n:
                        flag = False
                        break
                
            for prefix in self.not_trainable_prefixes:
                if prefix in n:
                    flag = True
                    break

            if flag:
                p.requires_grad_(False)
            else:
                print_rank0(n)
                p.requires_grad_(True)
                total_trainable += p.numel()
                assert p.requires_grad

        print_rank0("***** Total trainable parameters: " + str(total_trainable) + " *****")
    
    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass


    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def forward(self, x, batch):
        # self.ref_model.to(x.device)

        loss = self.loss_fn(self.model, self.ref_model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}

        # self.ref_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return loss_mean, loss_dict

    def add_noise_to_first_frame(self, image):
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(self.device)
        sigma = torch.exp(sigma).to(image.dtype)
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image = image + image_noise
        return image
    
    def add_mask_to_first_frame(self, image, mask_rate=0.25):
        b, c, f, h, w = image.shape
        image = image.permute(0, 2, 1, 3, 4).contiguous()
        rand_mask = torch.rand(h, w).to(dtype=image.dtype, device=image.device)
        mask = rand_mask > mask_rate
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
        mask = mask.expand(b, f, c, h, w)  

        image = image * mask
        
        image = image.permute(0, 2, 1, 3, 4).contiguous()
        return image

    def shared_step(self, batch: Dict, mask_rate=0.1) -> Any:
        x = self.get_input(batch)

        if self.lr_scale is not None:
            lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
            lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
            lr_z = self.encode_first_stage(lr_x, batch)
            batch["lr_input"] = lr_z
        
        ref_image = batch["ref_image"].to(dtype=self.dtype, device=self.device)
        ref_image = ref_image.permute(0, 2, 1, 3, 4).contiguous()
        
        mask_ref = batch["mask_ref"].to(dtype=self.dtype, device=self.device)
        mask_ref = mask_ref.permute(0, 2, 1, 3, 4).contiguous()
        mask_ref = self.encode_first_stage(mask_ref, None)
        mask_ref = mask_ref.permute(0, 2, 1, 3, 4).contiguous()
        
        
        n_motion_frame = 2
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if self.noised_image_input:
            image = x[:, :, 0:n_motion_frame]    # n_motion_frame=2
            image = self.add_noise_to_first_frame(image)
            image = self.add_mask_to_first_frame(image, mask_rate=mask_rate)
            image = torch.cat([ref_image, image], dim=2)
            image = self.encode_first_stage(image, batch)

        x = self.encode_first_stage(x, batch)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if self.noised_image_input:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
            
            assert self.noised_image_all_concat==False
            if self.noised_image_all_concat:
                image = image.repeat(1, x.shape[1], 1, 1, 1)
            else:
                image = torch.concat([image, torch.zeros_like(x[:, 1:])], dim=1)
            if random.random() < self.noised_image_dropout:
                image = torch.zeros_like(image)
            batch["concat_images"] = image

        
        if random.random() < 0.05:
            mask_ref = torch.zeros_like(mask_ref)
        batch["mask_ref"] = mask_ref
        
        gc.collect()
        torch.cuda.empty_cache()
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def get_input(self, batch):
        return batch[self.input_key].to(self.dtype)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples], **kwargs)
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, batch):
        frame = x.shape[2]

        if frame > 1 and self.latent_input:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            return x * self.scale_factor  # already encoded

        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        prefix=None,
        concat_images=None,
        audio_emb=None,
        ref_image=None,
        face_emb=None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)
        if hasattr(self, "seeded_noise"):
            randn = self.seeded_noise(randn)

        if prefix is not None:
            randn = torch.cat([prefix, randn[:, prefix.shape[1] :]], dim=1)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        if mp_size > 1:
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            torch.distributed.broadcast(randn, src=src, group=mpu.get_model_parallel_group())

        scale = None
        scale_emb = None
        
        ref_cond = deepcopy(cond)
        ref_cond['concat'] = ref_image
        ref_cond['crossattn'] = uc['crossattn']
        ref_timestep = torch.Tensor([0]).to(self.device)
        self.ref_model(ref_image, ref_timestep, ref_cond)
        num_layer = len(self.ref_model.diffusion_model.transformer.layers)
        for i in range(num_layer):
            self.model.diffusion_model.transformer.layers[i].bank = [
                torch.cat([v.clone()]*2, dim=0) 
                for v in self.ref_model.diffusion_model.transformer.layers[i].bank
            ]
            self.ref_model.diffusion_model.transformer.layers[i].bank.clear()

        denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
            self.model, input, sigma, c, concat_images=concat_images, **addtional_model_inputs
        )

        samples = self.sampler(denoiser, randn, cond, uc=uc, scale=scale, scale_emb=scale_emb, 
                               audio_emb=audio_emb, face_emb=face_emb)
        samples = samples.to(self.dtype)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if ((self.log_keys is None) or (embedder.input_key in self.log_keys)) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = ["x".join([str(xx) for xx in x[i].tolist()]) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_video(
        self,
        batch: Dict,
        N: int = 8,
        ucg_keys: List[str] = None,
        only_log_video_latents=False,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        if not self.latent_input:
            log["inputs"] = x.to(torch.float32)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.encode_first_stage(x, batch)
        if not only_log_video_latents:
            log["reconstructions"] = self.decode_first_stage(z).to(torch.float32)
            log["reconstructions"] = log["reconstructions"].permute(0, 2, 1, 3, 4).contiguous()
        z = z.permute(0, 2, 1, 3, 4).contiguous()

        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if self.noised_image_input:
            image = x[:, :, 0:1]
            image = self.add_noise_to_first_frame(image)
            image = self.encode_first_stage(image, batch)
            image = image.permute(0, 2, 1, 3, 4).contiguous()
            image = torch.concat([image, torch.zeros_like(z[:, 1:])], dim=1)
            c["concat"] = image
            uc["concat"] = image
            samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            if only_log_video_latents:
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                samples = self.decode_first_stage(samples).to(torch.float32)
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                log["samples"] = samples
        else:
            samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)  # b t c h w
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            if only_log_video_latents:
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                samples = self.decode_first_stage(samples).to(torch.float32)
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                log["samples"] = samples
        return log
