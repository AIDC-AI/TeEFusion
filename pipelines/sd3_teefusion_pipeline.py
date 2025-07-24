# Copyright (C) 2025 AIDC-AI
# This project is licensed under the Attribution-NonCommercial 4.0 International 
# License (SPDX-License-Identifier: CC-BY-NC-4.0). 

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.nn as nn

from typing import Union, List, Any
from diffusers.configuration_utils import ConfigMixin, register_to_config
from PIL import Image
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import DiffusionPipeline, AutoencoderKL
from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTokenizer, T5Tokenizer
from utils import encode_text, get_noise


class TeEFusionSD3Pipeline(DiffusionPipeline, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        transformer: nn.Module,
        text_encoder: CLIPTextModelWithProjection,
        text_encoder_2: CLIPTextModelWithProjection,
        text_encoder_3: T5EncoderModel,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        tokenizer_3: T5Tokenizer,
        vae: AutoencoderKL,
        scheduler: Any
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            vae=vae,
            scheduler=scheduler
        )

    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "TeEFusionSD3Pipeline":

        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        super().save_pretrained(save_directory)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: torch.FloatTensor = None,
        height: int = 1024,
        width: int = 1024,
        seed: int = 0,
    ):  
        if isinstance(prompt, str):
            prompt = [prompt]

        device = self.transformer.device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        prompt_embeds, pooled_prompt_embeds = encode_text(clip_tokenizers, clip_text_encoders, self.tokenizer_3, self.text_encoder_3, prompt, device)

        _, negative_pooled_prompt_embeds = encode_text(clip_tokenizers, clip_text_encoders, self.tokenizer_3, self.text_encoder_3, [''], device)


        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        bs = len(prompt)
        channels = self.transformer.config.in_channels
        height = 16 * (height // 16)
        width = 16 * (width // 16)

        # prepare input
        if latents is None:
            latents = get_noise(
                bs,
                channels,
                height,
                width,
                device=device,
                dtype=self.transformer.dtype,
                seed=seed,
            )

        for i, t in enumerate(timesteps):
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=t.reshape(1),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
                txt_align_guidance=torch.tensor(data=(guidance_scale,), dtype=self.transformer.dtype, device=self.transformer.device) * 1000.,
                txt_align_vec=pooled_prompt_embeds - negative_pooled_prompt_embeds 
            )[0]

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        x = latents.float()

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor is not None:
                    x = x / self.vae.config.scaling_factor
                if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor is not None:
                    x = x + self.vae.config.shift_factor
                x = self.vae.decode(x, return_dict=False)[0]

        # bring into PIL format and save
        x = (x / 2 + 0.5).clamp(0, 1)
        x = x.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (x * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images


    
