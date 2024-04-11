import os
from tqdm.auto import tqdm

try:
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler, 
        EulerDiscreteScheduler, 
        EulerAncestralDiscreteScheduler, 
        AutoencoderKL, 
        LCMScheduler,
        DDPMScheduler, 
        DEISMultistepScheduler, 
        PNDMScheduler,
        UniPCMultistepScheduler
)
    from diffusers.loaders.single_file_utils import (
        convert_ldm_vae_checkpoint, 
        convert_ldm_unet_checkpoint, 
        create_vae_diffusers_config, 
        create_unet_diffusers_config
    )
except:
    raise ImportError("Diffusers version too old. Please update to 0.26.0 minimum.")

import torch
from contextlib import nullcontext
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer, T5EncoderModel
from omegaconf import OmegaConf
from .modules.lora import monkeypatch_or_replace_lora_extended
from .modules.adapters import TextAdapter

import folder_paths
import comfy.latent_formats
import comfy.model_management as mm

script_directory = os.path.dirname(os.path.abspath(__file__))

class lavibridge_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "vae": ("VAE",),
            "lora_type": (
                [
                    'llama2_unet',
                    't5_unet',
                ], {
                    "default": 't5_unet'
                }),
            
            },
        }

    RETURN_TYPES = ("LAVIBRIDGE",)
    RETURN_NAMES = ("lavibridge",)
    FUNCTION = "loadmodel"
    CATEGORY = "LaVI-BridgeWrapper"

    def loadmodel(self, model, vae, lora_type):
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        vae_dtype = mm.vae_dtype()
        custom_config = {
            'model': model,
            'vae': vae,
        }
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            pbar = comfy.utils.ProgressBar(5)
            self.current_config = custom_config
            # config paths
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))
            
            # load models
            lavibridge_folder = os.path.join(folder_paths.models_dir,'lavibridge')
            lora_vis_path = os.path.join(lavibridge_folder, lora_type, 'lora_vis.pt')

            if not os.path.exists(lora_vis_path):
                print(f"Downloading LaVi-Bridge from https://huggingface.co/shihaozhao/LaVi-Bridge {lavibridge_folder}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="shihaozhao/LaVi-Bridge", allow_patterns=[f"*{lora_type}*"],local_dir=lavibridge_folder, local_dir_use_symlinks=False)
            print(f"Loaded LaVi-Bridge lora {lora_vis_path}")
            pbar.update(1)
            
            # get state dict from comfy models
            load_models = [model]
            comfy.model_management.load_models_gpu(load_models)
            sd = model.model.state_dict_for_saving(None, vae.get_sd(), None)

            pbar.update(1)

            # 1. vae
            converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
            vae = AutoencoderKL(**converted_vae_config)
            vae.load_state_dict(converted_vae, strict=False)
            vae.to(vae_dtype).eval()
            pbar.update(1)

            # 2. unet
            converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
            converted_unet = convert_ldm_unet_checkpoint(sd, converted_unet_config)
            unet = UNet2DConditionModel(**converted_unet_config)
            unet.load_state_dict(converted_unet, strict=False)
            unet.eval()
            pbar.update(1)

            # LoRA
            monkeypatch_or_replace_lora_extended(
                unet, 
                torch.load(lora_vis_path), 
                r=32, 
                target_replace_module={"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"},
            )
            unet.to(dtype)
           
            pbar.update(1)
            
            lavibridge_model = {
                'unet': unet,
                'vae': vae,
            }
   
        return (lavibridge_model,)


class lavi_bridge_llama_encoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": "Oppenheimer sits on the beach on a chair, watching a nuclear exposition with a huge mushroom cloud, 120mm",}),
            "max_length": ("INT", {"default": 77, "min": 1, "max": 512, "step": 1}),
            },
        }

    RETURN_TYPES = ("LAVIEMBEDS",)
    RETURN_NAMES = ("lavi_embeds",)
    FUNCTION = "process"
    CATEGORY = "LaVI-BridgeWrapper"

    def process(self, prompt, max_length):
        from transformers import LlamaForCausalLM, LlamaTokenizer
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        if not hasattr(self, "text_encoder"):    
            #llama2
            llama2_path = os.path.join(folder_paths.models_dir,'llama2', 'Llama-2-7b-hf')
            if not os.path.exists(llama2_path): 
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="NousResearch/Llama-2-7b-hf", local_dir=llama2_path,  ignore_patterns=["*.bin"], local_dir_use_symlinks=False)

            #adapter
            adapter_folder = os.path.join(folder_paths.models_dir,'lavibridge')
            adapter_path = os.path.join(adapter_folder, 'llama2_unet','adapter')
            if not os.path.exists(adapter_path):
                print(f"Downloading LaVi-Bridge from https://huggingface.co/shihaozhao/LaVi-Bridge {adapter_folder}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="shihaozhao/LaVi-Bridge", allow_patterns=["*llama2_unet*"],local_dir=adapter_folder, local_dir_use_symlinks=False)

            lora_text_path = os.path.join(adapter_folder, 'llama2_unet', 'lora_text.pt')

            self.adapter = TextAdapter.from_pretrained(adapter_path).eval().to(dtype)
            self.tokenizer = LlamaTokenizer.from_pretrained(llama2_path)
            self.tokenizer.pad_token = '[PAD]'
            self.text_encoder = LlamaForCausalLM.from_pretrained(llama2_path, torch_dtype=dtype)
           
            monkeypatch_or_replace_lora_extended(
                self.text_encoder, 
                torch.load(lora_text_path), 
                r=32, 
                target_replace_module = {"LlamaAttention"},
            )

        self.adapter.to(device)
        self.text_encoder.to(device)

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            text_ids = self.tokenizer(prompt, padding="max_length", max_length=max_length, return_tensors="pt", truncation=True).input_ids.to(device)
            text_embeddings = self.text_encoder(input_ids=text_ids, output_hidden_states=True).hidden_states[-1]
            text_embeddings = self.adapter(text_embeddings).sample
            uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device), output_hidden_states=True).hidden_states[-1]
            uncond_embeddings =  self.adapter(uncond_embeddings).sample
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            self.adapter.to(offload_device)
            self.text_encoder.to(offload_device)

            return (text_embeddings,)
        
class lavi_bridge_t5_encoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": "Oppenheimer sits on the beach on a chair, watching a nuclear exposition with a huge mushroom cloud, 120mm",}),
            "max_length": ("INT", {"default": 77, "min": 1, "max": 512, "step": 1}),
            },
        }

    RETURN_TYPES = ("LAVIEMBEDS",)
    RETURN_NAMES = ("lavi_embeds",)
    FUNCTION = "process"
    CATEGORY = "LaVI-BridgeWrapper"

    def process(self, prompt, max_length):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()
        #dtype = mm.unet_dtype()
        dtype = torch.bfloat16
        if not hasattr(self, "text_encoder"):    
            #t5
            t5_path = os.path.join(folder_paths.models_dir,'t5_model', 't5-large-encoder-only-bf16')
            if not os.path.exists(t5_path): 
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="Kijai/t5-large-encoder-only-bf16", local_dir=t5_path, local_dir_use_symlinks=False)

            #adapter
            adapter_folder = os.path.join(folder_paths.models_dir,'lavibridge')
            adapter_path = os.path.join(adapter_folder, 't5_unet','adapter')
            if not os.path.exists(adapter_path):
                print(f"Downloading LaVi-Bridge from https://huggingface.co/shihaozhao/LaVi-Bridge {adapter_folder}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="shihaozhao/LaVi-Bridge", allow_patterns=["*t5_unet*"],local_dir=adapter_folder, local_dir_use_symlinks=False)

            lora_text_path = os.path.join(adapter_folder, 't5_unet', 'lora_text.pt')

            self.adapter = TextAdapter.from_pretrained(adapter_path).eval().to(dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(t5_path)
            self.text_encoder = T5EncoderModel.from_pretrained(t5_path).eval().to(dtype)

            monkeypatch_or_replace_lora_extended(
                self.text_encoder, 
                torch.load(lora_text_path), 
                r=32, 
                target_replace_module = {"T5Attention"},
            )

        self.adapter.to(device)
        self.text_encoder.to(device)

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            text_ids = self.tokenizer(prompt, padding="max_length", max_length=max_length, return_tensors="pt", truncation=True).input_ids.to(device)
            text_embeddings = self.text_encoder(input_ids=text_ids)[0]
            text_embeddings = self.adapter(text_embeddings).sample
            uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
            uncond_embeddings =  self.adapter(uncond_embeddings).sample
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            self.adapter.to(offload_device)
            self.text_encoder.to(offload_device)

            return (text_embeddings,)
        
class lavibridge_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "lavibridge_model": ("LAVIBRIDGE",),
            "lavi_embeds": ("LAVIEMBEDS",),
            "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "scheduler": (
                [
                    'DPMSolverMultistepScheduler',
                    'DPMSolverMultistepScheduler_SDE_karras',
                    'DDPMScheduler',
                    'LCMScheduler',
                    'PNDMScheduler',
                    'DEISMultistepScheduler',
                    'EulerDiscreteScheduler',
                    'EulerAncestralDiscreteScheduler',
                    'UniPCMultistepScheduler',
                    'DDIMScheduler',
                ], {
                    "default": 'DPMSolverMultistepScheduler'
                }),
            },    
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "LaVI-BridgeWrapper"

    def process(self, lavibridge_model, lavi_embeds, width, height, batch_size, steps, guidance_scale, seed, scheduler):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        torch.manual_seed(seed)
        dtype = mm.unet_dtype()

        unet = lavibridge_model["unet"]
        vae = lavibridge_model["vae"]

        scheduler_config = {
                'num_train_timesteps': 1000,
                'beta_start':    0.00085,
                'beta_end':      0.012,
                'beta_schedule': "scaled_linear",
                'steps_offset': 1,
            }
        if scheduler == 'DPMSolverMultistepScheduler':
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DPMSolverMultistepScheduler_SDE_karras':
            scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
            scheduler_config.update({"use_karras_sigmas": True})
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDPMScheduler':
            noise_scheduler = DDPMScheduler(**scheduler_config)
        elif scheduler == 'LCMScheduler':
            noise_scheduler = LCMScheduler(**scheduler_config)
        elif scheduler == 'PNDMScheduler':
            scheduler_config.update({"set_alpha_to_one": False})
            scheduler_config.update({"trained_betas": None})
            noise_scheduler = PNDMScheduler(**scheduler_config)
        elif scheduler == 'DEISMultistepScheduler':
            noise_scheduler = DEISMultistepScheduler(**scheduler_config)
        elif scheduler == 'EulerDiscreteScheduler':
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == 'EulerAncestralDiscreteScheduler':
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        elif scheduler == 'UniPCMultistepScheduler':
            noise_scheduler = UniPCMultistepScheduler(**scheduler_config)
        elif scheduler == 'DDIMScheduler':
            noise_scheduler = DDIMScheduler(**scheduler_config)

        unet.to(device)

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            # Latent preparation
            vae.to(device)
            latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8)).to(device)
            latents = latents * noise_scheduler.init_noise_sigma
            vae.to(offload_device)

            lavi_embeds_repeated = lavi_embeds.repeat_interleave(batch_size, dim=0)
            # Model prediction
            noise_scheduler.set_timesteps(steps)

            for t in tqdm(noise_scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2, dim=0)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=lavi_embeds_repeated).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            unet.to(offload_device)

            # Decoding
            vae.to(device)
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample
            vae.to(offload_device)

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.permute(0, 2, 3, 1).cpu().float()
            return (image,)

NODE_CLASS_MAPPINGS = {
    "lavibridge_sampler": lavibridge_sampler,
    "lavi_bridge_t5_encoder": lavi_bridge_t5_encoder,
    "lavibridge_model_loader": lavibridge_model_loader,
    "lavi_bridge_llama_encoder": lavi_bridge_llama_encoder
  
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "lavibridge_sampler": "LaVi-Bridge Sampler",
    "lavi_bridge_t5_encoder": "LaVi-Bridge T5 Encoder",
    "lavibridge_model_loader": "LaVi-Bridge Model Loader",
    "lavi_bridge_llama_encoder": "LaVi-Bridge LLaMA Encoder"
}
