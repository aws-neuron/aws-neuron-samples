import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2" # Comment this line out if using trn1/inf2
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2" # Comment this line out if using trn1/inf2
compiler_flags = """ --verbose=INFO --target=trn2 --lnc=2 --internal-hlo2tensorizer-options='--fuse-dot-logistic=false' --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn2
# compiler_flags = """ --verbose=INFO --target=trn1 --model-type=unet-inference --enable-fast-loading-neuron-binaries """ # Use these compiler flags for trn1/inf2
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

from diffusers import PixArtSigmaPipeline
import torch
import argparse
import neuronx_distributed

from torch import nn
from functools import partial

from neuron_commons import attention_wrapper_for_transformer
from neuron_parallel_utils import shard_transformer_attn, shard_transformer_feedforward

from diffusers.models.transformers.pixart_transformer_2d import PixArtTransformer2DModel
torch.nn.functional.scaled_dot_product_attention = attention_wrapper_for_transformer

class TracingTransformerWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device    
    
    def forward(self, hidden_states=None, encoder_hidden_states=None, timestep=None, encoder_attention_mask=None, **kwargs):
        return self.transformer(
        hidden_states=hidden_states, 
        encoder_hidden_states=encoder_hidden_states, 
        timestep=timestep, 
        encoder_attention_mask=encoder_attention_mask,
        added_cond_kwargs={"resolution": None, "aspect_ratio": None},
        return_dict=False)

def get_transformer_model(tp_degree: int):
    pipe: PixArtSigmaPipeline = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir="pixart_sigma_hf_cache_dir_1024")
    # 28 of these.
    for block_idx, block in enumerate(pipe.transformer.transformer_blocks):
        block.attn1 = shard_transformer_attn(tp_degree, block.attn1)
        block.attn2 = shard_transformer_attn(tp_degree, block.attn2)
        block.ff = shard_transformer_feedforward(block.ff)

    mod_pipe_transformer_f = TracingTransformerWrapper(pipe.transformer)
    return mod_pipe_transformer_f, {}

def compile_transformer(args):
    tp_degree = 4
    # tp_degree = 8 # Use tensor parallel degree as 8 for trn1/inf2
    os.environ["LOCAL_WORLD_SIZE"] = "4" # Use tensor parallel degree as 4 for trn2
    latent_height = args.height//8
    latent_width = args.width//8
    num_prompts = 1
    num_images_per_prompt = args.num_images_per_prompt
    max_sequence_length = args.max_sequence_length
    hidden_size = 4096
    compiler_workdir = args.compiler_workdir
    compiled_models_dir = args.compiled_models_dir
    batch_size = 2
    sample_hidden_states = torch.ones((batch_size, 4, latent_height, latent_width), dtype=torch.bfloat16)
    sample_encoder_hidden_states = torch.ones((batch_size, max_sequence_length, hidden_size), dtype=torch.bfloat16)
    sample_timestep = torch.ones((batch_size), dtype=torch.int64)
    sample_encoder_attention_mask = torch.ones((batch_size, max_sequence_length), dtype=torch.int64)

    get_transformer_model_f = partial(get_transformer_model, tp_degree)
    with torch.no_grad():
        sample_inputs = sample_hidden_states, sample_encoder_hidden_states, sample_timestep, sample_encoder_attention_mask
        compiled_transformer = neuronx_distributed.trace.parallel_model_trace(
            get_transformer_model_f,
            sample_inputs,
            compiler_workdir=f"{compiler_workdir}/transformer",
            compiler_args=compiler_flags,
            tp_degree=tp_degree,
            inline_weights_to_neff=False,
        )
        compiled_model_dir = f"{compiled_models_dir}/transformer"
        if not os.path.exists(compiled_model_dir):
            os.makedirs(compiled_model_dir)         
        neuronx_distributed.trace.parallel_model_save(
            compiled_transformer, f"{compiled_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", help="height of generated image.", type=int, default=1024)
    parser.add_argument("--width", help="width of generated image.", type=int, default=1024)
    parser.add_argument("--num_images_per_prompt", help="number of images per prompt.", type=int, default=1)
    parser.add_argument("--max_sequence_length", help="max sequence length.", type=int, default=300)
    parser.add_argument("--compiler_workdir", help="dir for compiler artifacts.", type=str, default="compiler_workdir")
    parser.add_argument("--compiled_models_dir", help="dir for compiled artifacts.", type=str, default="compiled_models")
    args = parser.parse_args()
    compile_transformer(args)