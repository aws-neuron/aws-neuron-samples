import torch
from diffusers import PixArtSigmaPipeline

pipe: PixArtSigmaPipeline = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    torch_dtype=torch.bfloat16,
    cache_dir="pixart_sigma_hf_cache_dir_1024")