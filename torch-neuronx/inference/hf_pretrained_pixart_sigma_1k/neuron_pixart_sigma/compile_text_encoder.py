import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2" # Comment this line out if using trn1/inf2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2" # Comment this line out if using trn1/inf2
compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn2
# compiler_flags = """ --verbose=INFO --target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import PixArtSigmaPipeline
import torch
import argparse
import torch_neuronx
import neuronx_distributed
from transformers.models.t5 import T5EncoderModel
from torch import nn
from functools import partial

from transformers.models.t5.modeling_t5 import T5EncoderModel, T5Block, T5LayerSelfAttention, T5LayerFF

from neuron_commons import attention_wrapper, f32Wrapper
from neuron_parallel_utils import get_sharded_data, shard_t5_self_attention, shard_t5_ff

torch.nn.functional.scaled_dot_product_attention = attention_wrapper


class TracingT5WrapperTP(nn.Module):
    def __init__(self, t: T5EncoderModel, seqlen: int):
        super().__init__()
        self.t = t
        self.device = t.device
        precomputed_bias = self.t.encoder.block[0].layer[0].SelfAttention.compute_bias(seqlen, seqlen)
        precomputed_bias_tp = get_sharded_data(precomputed_bias, 1)
        self.t.encoder.block[0].layer[0].SelfAttention.compute_bias = lambda *args, **kwargs: precomputed_bias_tp
    
    def forward(self, text_input_ids, prompt_attention_mask):
        return self.t(
            text_input_ids, 
            attention_mask=prompt_attention_mask
        )

def get_text_encoder(tp_degree: int, sequence_length: int):
    pipe: PixArtSigmaPipeline = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        cache_dir="pixart_sigma_hf_cache_dir_1024",
        local_files_only=True,
        torch_dtype=torch.bfloat16)
    text_encoder: T5EncoderModel = pipe.text_encoder
    text_encoder.eval()
    for idx, block in enumerate(text_encoder.encoder.block):
        block: T5Block = block
        block.layer[1].DenseReluDense.act = torch.nn.GELU(approximate="tanh")
        selfAttention: T5LayerSelfAttention = block.layer[0].SelfAttention
        ff: T5LayerFF = block.layer[1]
        layer_norm_0 = block.layer[0].layer_norm.to(torch.float32)
        layer_norm_1 = block.layer[1].layer_norm.to(torch.float32)       
        block.layer[1] = shard_t5_ff(ff)
        block.layer[0].SelfAttention = shard_t5_self_attention(tp_degree, selfAttention)
        block.layer[0].layer_norm = f32Wrapper(layer_norm_0)
        block.layer[1].layer_norm = f32Wrapper(layer_norm_1)
    final_layer_norm = pipe.text_encoder.encoder.final_layer_norm.to(torch.float32)
    pipe.text_encoder.encoder.final_layer_norm = f32Wrapper(final_layer_norm)             
    return TracingT5WrapperTP(text_encoder, sequence_length), {}

def compile_text_encoder(args):
    batch_size = 1 # batch_size = args.num_prompts
    sequence_length = args.max_sequence_length
    tp_degree = 4 # Use tensor parallel degree as 4 for trn2
    # tp_degree = 8 # Use tensor parallel degree as 8 for trn1/inf2
    os.environ["LOCAL_WORLD_SIZE"] = "4"
    get_text_encoder_f = partial(get_text_encoder, tp_degree, sequence_length)
    
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    
    with torch.no_grad():
        sample_inputs = torch.ones((batch_size, sequence_length), dtype=torch.int64), \
            torch.ones((batch_size, sequence_length), dtype=torch.int64)
        compiled_text_encoder = neuronx_distributed.trace.parallel_model_trace(
            get_text_encoder_f,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/text_encoder",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
        )
        compiled_model_dir = f"{compiled_models_dir}/text_encoder"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)           
        neuronx_distributed.trace.parallel_model_save(
            compiled_text_encoder, f"{compiled_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", help="number of prompts", type=int, default=1)
    parser.add_argument("--max_sequence_length", help="max sequence length.", type=int, default=300)
    parser.add_argument("--compiler_workdir", help="dir for compiler artifacts.", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts.", type=str,  default="compiled_models")
    args = parser.parse_args()
    compile_text_encoder(args)