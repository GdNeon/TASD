# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from typing import Optional, Union, Tuple, Dict
from PIL import Image
from einops import rearrange
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

def save_images(images,dest, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    pil_img = Image.fromarray(images[-1])
    pil_img.save(dest)
    # display(pil_img)

def save_image(images,dest, num_rows=1, offset_ratio=0.02):
    print(images.shape)
    pil_img = Image.fromarray(images[0])
    pil_img.save(dest)

def reshape_heads_to_batch_dim(head_size, tensor):
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
    return tensor

def reshape_batch_dim_to_heads(head_size, tensor):
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    return tensor

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

def register_attention_control(model, controller, ip_scale):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out #linear网络
        def forward(x, encoder_hidden_states=None, attention_mask=None):
            h = self.heads
            q = self.to_q(x)
            encoder_hidden_states = x
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            q = reshape_heads_to_batch_dim(h, q)
            k = reshape_heads_to_batch_dim(h, k)
            v = reshape_heads_to_batch_dim(h, v)
            q, k, v = controller.self_attn_forward(q, k, v, h)
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = reshape_batch_dim_to_heads(h, out)
            out = to_out[0](out)
            out = to_out[1](out) / self.rescale_output_factor
            return out 
        return forward
    
    def ca_forward_ip(self, place_in_unet):
        to_out = self.to_out #linear网络
        def forward(x, encoder_hidden_states=None, attention_mask=None):
            h = self.heads
            q = self.to_q(x)
            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = x
            else:
                end_pos = encoder_hidden_states.shape[1] - 4
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    encoder_hidden_states[:, end_pos:, :],
                )
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)
            q = reshape_heads_to_batch_dim(h, q)
            k = reshape_heads_to_batch_dim(h, k)
            v = reshape_heads_to_batch_dim(h, v)
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet) #只保存没有ip部分的attn_map
            hidden_states = torch.einsum("b i j, b j d -> b i d", attn, v)
            hidden_states = reshape_batch_dim_to_heads(h, hidden_states)
            ip_k = self.processor.to_k_ip(ip_hidden_states)
            ip_v = self.processor.to_v_ip(ip_hidden_states)
            ip_k = reshape_heads_to_batch_dim(h, ip_k)
            ip_v = reshape_heads_to_batch_dim(h, ip_v)
            sim = torch.einsum("b i d, b j d -> b i j", q, ip_k) * self.scale
            attn = sim.softmax(dim=-1)
            ip_hidden_states = torch.einsum("b i j, b j d -> b i d", attn, ip_v)
            ip_hidden_states = reshape_batch_dim_to_heads(h, ip_hidden_states)
            out = hidden_states + ip_scale * ip_hidden_states #可以对hidden_states做一个mask 
            out = to_out[0](out)
            out = to_out[1](out) / self.rescale_output_factor                   
            return out
        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            if net_.processor.__class__.__name__ == 'AttnProcessor2_0':  
                net_.forward = ca_forward(net_, place_in_unet)
            else:
                net_.forward = ca_forward_ip(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words) # time, batch, heads, pixels, words
    return alpha_time_words
