""" NxD GPTNeoX model """

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

from neuronx_distributed.parallel_layers.layers import ParallelEmbedding, ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size
import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils
from neuronx_distributed.parallel_layers import move_model_to_device, mappings, layer_norm
import torch_xla.core.xla_model as xm

from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXMLP,
    GPTNeoXLayer,
    GPTNeoXModel,
    GPTNeoXPreTrainedModel,
    GPTNeoXForCausalLM,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)

from utils import scatter_to_sequence_parallel_region

from functools import partial
def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)

logger = logging.get_logger(__name__)


class GPTNeoXAttentionNxD(GPTNeoXAttention):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())

        # NxD code change: Replace the Linear with ColumnParallelLinear and RowParallelLinear
        self.config = config
        self.num_attention_heads = neuronx_dist_utils.divide(config.num_attention_heads, get_tensor_model_parallel_size()) 
        init_method = partial(_init_normal, config.initializer_range)
        self.query_key_value = ColumnParallelLinear(
            config.hidden_size,
            3 * config.hidden_size,
            stride=3,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )
        self.dense = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )
        with torch.no_grad():
            self.query_key_value.bias.data.zero_()
            self.dense.bias.data.zero_()
        move_model_to_device(self, xm.xla_device())

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # NxD code change: sequence parallel uses seq_len as the 0-th dim
        if self.config.sequence_parallel_enabled:
            # [seq_len, batch, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
            query = qkv[..., : self.head_size].permute(1, 2, 0, 3)
            key = qkv[..., self.head_size : 2 * self.head_size].permute(1, 2, 0, 3)
            value = qkv[..., 2 * self.head_size :].permute(1, 2, 0, 3)
        else:
            # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
            query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
            key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
            value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        # NxD code change: sequence parallel uses seq_len as the 0-th dim
        if self.config.sequence_parallel_enabled:
            # tensor [bs, num_attention_heads, seq_len, attn_head_size]
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
            # -> [seq_len, bs, num_attention_heads, attn_head_size]
            # -> [seq_len, bs hidden_size]
        else:
            # tensor [bs, num_attention_heads, seq_len, attn_head_size]
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
            # -> [bs, seq_len, num_attention_heads, attn_head_size]
            # -> [bs, seq_len, hidden_size]
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), self.num_attention_heads * self.head_size)

        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        # NxD code change: creating causal_mask on-the-fly to trigger the Neuron compiler optimization
        causal_mask = torch.triu(torch.ones((1, 1, query_length, key_length), device='xla'), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill_(causal_mask, -10000.0)

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class GPTNeoXMLPNxD(GPTNeoXMLP):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.act = ACT2FN[config.hidden_act]

        # NxD code change: Replace the Linear with ColumnParallelLinear and RowParallelLinear
        self.config = config
        init_method = partial(_init_normal, config.initializer_range)
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )
        self.dense_4h_to_h = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )
        with torch.no_grad():
            self.dense_h_to_4h.bias.data.zero_()
            self.dense_4h_to_h.bias.data.zero_()
        move_model_to_device(self, xm.xla_device())


class GPTNeoXLayerNxD(GPTNeoXLayer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.use_parallel_residual = config.use_parallel_residual

        # NxD code change: Replace the nn LayerNorm with nxd LayerNorm to use sequence parallel
        self.input_layernorm = layer_norm.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, sequence_parallel_enabled=config.sequence_parallel_enabled)
        self.input_layernorm.bias.data.zero_()
        self.input_layernorm.weight.data.fill_(1.0)
        self.post_attention_layernorm = layer_norm.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, sequence_parallel_enabled=config.sequence_parallel_enabled)
        self.post_attention_layernorm.bias.data.zero_()
        self.post_attention_layernorm.weight.data.fill_(1.0)

        self.attention = GPTNeoXAttentionNxD(config)
        self.mlp = GPTNeoXMLPNxD(config)


class GPTNeoXModelNxD(GPTNeoXModel):
    def __init__(self, config):
        GPTNeoXPreTrainedModel.__init__(self, config)
        self.config = config

        # NxD code change: Replace the Embedding with ParallelEmbedding
        init_method = partial(_init_normal, config.initializer_range)
        self.embed_in = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                init_method=init_method,
        )

        self.layers = nn.ModuleList([GPTNeoXLayerNxD(config) for _ in range(config.num_hidden_layers)])

        # Replace the nn LayerNorm with nxd LayerNorm to use sequence parallel
        self.final_layer_norm = layer_norm.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, sequence_parallel_enabled=config.sequence_parallel_enabled)
        self.final_layer_norm.bias.data.zero_()
        self.final_layer_norm.weight.data.fill_(1.0)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        hidden_states = inputs_embeds

        # NxD code change: sequence parallel uses seq_len as the 0-th dim
        if self.config.sequence_parallel_enabled:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for layer_past
                        return module(*inputs, use_cache, None, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)

        # NxD code change: sequence parallel uses seq_len as the 0-th dim
        if self.config.sequence_parallel_enabled:
            hidden_states = mappings.gather_from_sequence_parallel_region(hidden_states, to_model_parallel=False)
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class GPTNeoXForCausalLMNxD(GPTNeoXForCausalLM):

    def __init__(self, config):
        GPTNeoXPreTrainedModel.__init__(self, config)

        self.gpt_neox = GPTNeoXModelNxD(config)

        # NxD code change: Replace the Linear with ColumnParallelLinear
        init_method = partial(_init_normal, config.initializer_range)
        self.embed_out = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
        >>> config.is_decoder = True
        >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            # NxD code change: Replace the CrossEntropyLoss with parallel_cross_entropy
            loss_fct = parallel_cross_entropy

            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

            # NxD code change: parallel_cross_entropy requires to take an averge
            lm_loss = torch.mean(lm_loss)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
