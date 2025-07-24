# Copyright (C) 2025 AIDC-AI
# This project is licensed under the Attribution-NonCommercial 4.0 International 
# License (SPDX-License-Identifier: CC-BY-NC-4.0). 

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import List, Optional, Union

def get_noise(
    num_samples: int,
    channel: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        channel,
        height // 8,
        width // 8,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

def get_clip_prompt_embeds(
    clip_tokenizers, 
    clip_text_encoders,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    clip_skip: Optional[int] = None,
    clip_model_index: int = 0,
):

    tokenizer_max_length = 77
    tokenizer = clip_tokenizers[clip_model_index]
    text_encoder = clip_text_encoders[clip_model_index]

    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]

    if clip_skip is None:
        prompt_embeds = prompt_embeds.hidden_states[-2]
    else:
        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds, pooled_prompt_embeds
    
def get_t5_prompt_embeds(
    tokenizer_3,
    text_encoder_3,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 256,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):

    tokenizer_max_length = 77
    batch_size = len(prompt)

    text_inputs = tokenizer_3(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer_3.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])

    prompt_embeds = text_encoder_3(text_input_ids.to(device))[0]

    dtype = text_encoder_3.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

    
@torch.no_grad()
def encode_text(clip_tokenizers, clip_text_encoders, tokenizer_3, text_encoder_3, prompt, device, max_sequence_length=256):
    
    prompt_embed, pooled_prompt_embed = get_clip_prompt_embeds(clip_tokenizers, clip_text_encoders, prompt=prompt, device=device, clip_model_index=0)
    prompt_2_embed, pooled_prompt_2_embed = get_clip_prompt_embeds(clip_tokenizers, clip_text_encoders, prompt=prompt, device=device, clip_model_index=1)
    clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

    t5_prompt_embed = get_t5_prompt_embeds(tokenizer_3, text_encoder_3, prompt=prompt, max_sequence_length=max_sequence_length, device=device)

    clip_prompt_embeds = torch.nn.functional.pad(clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]))

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

    return prompt_embeds, pooled_prompt_embeds