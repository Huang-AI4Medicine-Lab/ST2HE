"""Minimal pix2pix-turbo backbone vendored for ST2HE inference."""

import copy

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

from .model import make_1step_sched, my_vae_decoder_fwd, my_vae_encoder_fwd


class TwinConv(torch.nn.Module):
    """Blend pretrained and current conv_in weights for stochastic generation."""

    def __init__(self, convin_pretrained, convin_curr):
        super().__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * self.r


class Pix2Pix_Turbo(torch.nn.Module):
    """Vendored pix2pix-turbo inference backbone used by ST2HE."""

    def __init__(
        self,
        pretrained_name=None,
        pretrained_path=None,
        lora_rank_unet=8,
        lora_rank_vae=4,
        device=None,
        enable_xformers=True,
    ):
        super().__init__()
        if pretrained_name is not None:
            raise ValueError(
                "The vendored ST2HE copy only supports a local pretrained weights file."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/sd-turbo",
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/sd-turbo",
            subfolder="text_encoder",
        ).to(self.device)
        self.sched = make_1step_sched(self.device)

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(
            vae.encoder,
            vae.encoder.__class__,
        )
        vae.decoder.forward = my_vae_decoder_fwd.__get__(
            vae.decoder,
            vae.decoder.__class__,
        )
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=1, bias=False)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=1, bias=False)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=1, bias=False)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=1, bias=False)
        vae.decoder.ignore_skip = False

        unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/sd-turbo",
            subfolder="unet",
        )

        if pretrained_path is not None:
            self._load_pretrained_weights(vae, unet, pretrained_path)
        else:
            self._init_random_adapters(vae, unet, lora_rank_unet, lora_rank_vae)

        if enable_xformers and hasattr(unet, "enable_xformers_memory_efficient_attention"):
            try:
                unet.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.unet = unet.to(self.device)
        self.vae = vae.to(self.device)
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device=self.device).long()
        self.text_encoder.requires_grad_(False)

    @staticmethod
    def _load_pretrained_weights(vae, unet, pretrained_path):
        sd = torch.load(pretrained_path, map_location="cpu")
        unet_lora_config = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["unet_lora_target_modules"],
        )
        vae_lora_config = LoraConfig(
            r=sd["rank_vae"],
            init_lora_weights="gaussian",
            target_modules=sd["vae_lora_target_modules"],
        )
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        vae_state_dict = vae.state_dict()
        for key, value in sd["state_dict_vae"].items():
            vae_state_dict[key] = value
        vae.load_state_dict(vae_state_dict)
        unet.add_adapter(unet_lora_config)
        unet_state_dict = unet.state_dict()
        for key, value in sd["state_dict_unet"].items():
            unet_state_dict[key] = value
        unet.load_state_dict(unet_state_dict)

    def _init_random_adapters(self, vae, unet, lora_rank_unet, lora_rank_vae):
        torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
        target_modules_vae = [
            "conv1",
            "conv2",
            "conv_in",
            "conv_shortcut",
            "conv",
            "conv_out",
            "skip_conv_1",
            "skip_conv_2",
            "skip_conv_3",
            "skip_conv_4",
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ]
        vae_lora_config = LoraConfig(
            r=lora_rank_vae,
            init_lora_weights="gaussian",
            target_modules=target_modules_vae,
        )
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        target_modules_unet = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "conv",
            "conv1",
            "conv2",
            "conv_shortcut",
            "conv_out",
            "proj_in",
            "proj_out",
            "ff.net.2",
            "ff.net.0.proj",
        ]
        unet_lora_config = LoraConfig(
            r=lora_rank_unet,
            init_lora_weights="gaussian",
            target_modules=target_modules_unet,
        )
        unet.add_adapter(unet_lora_config)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for name, param in self.vae.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(
        self,
        c_t,
        prompt=None,
        prompt_tokens=None,
        deterministic=True,
        r=1.0,
        noise_map=None,
    ):
        """Run the one-step pix2pix-turbo forward pass."""
        assert (prompt is None) != (
            prompt_tokens is None
        ), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            caption_tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens.to(self.device))[0]

        encoded_control = (
            self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        )
        if deterministic:
            model_pred = self.unet(
                encoded_control,
                self.timesteps,
                encoder_hidden_states=caption_enc,
            ).sample
            x_denoised = self.sched.step(
                model_pred,
                self.timesteps,
                encoded_control,
                return_dict=True,
            ).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (
                self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
            ).clamp(-1, 1)
            return output_image

        self.unet.set_adapters(["default"], weights=[r])
        set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
        if noise_map is None:
            raise ValueError("noise_map is required when deterministic=False")
        unet_input = encoded_control * r + noise_map * (1 - r)
        self.unet.conv_in.r = r
        unet_output = self.unet(
            unet_input,
            self.timesteps,
            encoder_hidden_states=caption_enc,
        ).sample
        self.unet.conv_in.r = None
        x_denoised = self.sched.step(
            unet_output,
            self.timesteps,
            unet_input,
            return_dict=True,
        ).prev_sample
        x_denoised = x_denoised.to(unet_output.dtype)
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        self.vae.decoder.gamma = r
        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)
        return output_image


Pix2PixTurbo = Pix2Pix_Turbo
