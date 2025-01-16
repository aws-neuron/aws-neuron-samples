from diffusers import PixArtSigmaPipeline, Transformer2DModel
from transformers.models.t5.modeling_t5 import T5EncoderModel
from torch import nn

class InferenceTextEncoderWrapper(nn.Module):
    def __init__(self, dtype, t: T5EncoderModel, seqlen: int):
        super().__init__()
        self.dtype = dtype
        self.device = t.device
        self.t = t
    def forward(self, text_input_ids, attention_mask=None):
        return [self.t(text_input_ids, attention_mask)['last_hidden_state'].to(self.dtype)]

class InferenceTransformerWrapper(nn.Module):
    def __init__(self, transformer: Transformer2DModel):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device
    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, 
                            encoder_attention_mask=None, added_cond_kwargs=None,
                            return_dict=False):
        output = self.transformer(
            hidden_states, 
            encoder_hidden_states, 
            timestep, 
            encoder_attention_mask)
        return output

class SimpleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        output = self.model(x)
        return output

import torch
import math
from torch import nn

from neuronxcc.starfish.penguin.targets.nki.private_api import vnc
from torch_neuronx.xla_impl.ops import nki_jit
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
_flash_fwd_call = nki_jit()(attention_isa_kernel)


def neuron_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
    orig_shape = None
    if len(query.shape) == 4:
        orig_shape = query.shape
        def to3d(x):
            return x.reshape(-1, x.shape[2], x.shape[3])
        query, key, value = map(to3d, [query, key, value])
    if query.size() == key.size():
        attention_scores = torch.bmm(key, query.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=1).permute(0, 2, 1)
    else:
        attention_scores = torch.bmm(query, key.transpose(-1, -2)) * (
            1 / math.sqrt(query.size(-1))
        )
        attention_probs = attention_scores.softmax(dim=-1)
    attn_out = torch.bmm(attention_probs, value)
    if orig_shape:
        attn_out = attn_out.reshape(
            orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
        )
    return attn_out


def attention_wrapper_sharded_without_swap(query, key, value):
    bs, n_head, q_len, d_head = query.shape
    q = query.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
    v = value.clone().reshape((bs*n_head, q_len, d_head))
    attn_output = torch.zeros((bs*n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)
    use_sharded_attention_kernel = True
    if use_sharded_attention_kernel:
        grid = (vnc(2),)
        _flash_fwd_call[grid](q, k, v, 0.117, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, 0.117, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
    return attn_output


sdpa_original = torch.nn.functional.scaled_dot_product_attention
def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
    if attn_mask is not None:
        return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    else:
        return neuron_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        
def attention_wrapper_for_transformer(query, key, value, attn_mask=None, dropout_p=None, is_causal=None):
    if attn_mask is not None:
        return sdpa_original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    else:
        return attention_wrapper_sharded_without_swap(query, key, value)
        
class f32Wrapper(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
    def forward(self, x):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y)
        return output.type(t)
    
    