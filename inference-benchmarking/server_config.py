import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ServerConfig:
    name: str
    model_path: str
    model_s3_path: str
    max_seq_len: int
    context_encoding_len: int
    tp_degree: int
    server_port: int
    continuous_batch_size: int = 1
    n_vllm_threads: int = 32

    # Optional configurations
    draft_model_path: Optional[str] = None
    draft_model_s3_path: Optional[str] = None
    sharded_weights_path: Optional[str] = None
    sharded_weights_s3_path: Optional[str] = None
    spec_len: Optional[int] = None
    speculation_type: Optional[str] = None
    compiled_model_path: Optional[str] = None
    inference_demo_script: Optional[str] = None
    inference_demo_args: Optional[str] = None
    scratchpad_page_size: Optional[int] = None
    enable_scratchpad_single_core_debugging: Optional[bool] = False
    custom_chat_template_path: Optional[str] = None
    override_neuron_config: Optional[Union[str, dict]] = None
    quantization_param_path: Optional[str] = None
    quant_dtype: Optional[str] = None

    def __post_init__(self):
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.context_encoding_len <= 0:
            raise ValueError("context_encoding_len must be positive")
        if self.tp_degree <= 0:
            raise ValueError("tp_degree must be positive")
        if self.n_vllm_threads <= 0:
            raise ValueError("n_vllm_threads must be positive")
        if self.continuous_batch_size <= 0:
            raise ValueError("continuous_batch_size must be positive")
        if self.server_port < 0 or self.server_port > 65535:
            raise ValueError("server_port must be between 0 and 65535")

        # Validate optional configurations
        if self.spec_len is not None and self.spec_len <= 0:
            raise ValueError("spec_len must be positive if specified")
        if self.speculation_type and self.speculation_type not in ["eagle"]:
            raise ValueError("speculation_type must be 'eagle' if specified")

        # parse override_neuron_config or initialize it as an empty dict
        if self.override_neuron_config:
            if isinstance(self.override_neuron_config, str):
                # Assume it's a path to a JSON/YAML file
                try:
                    import json

                    self.override_neuron_config = json.loads(self.override_neuron_config)
                except Exception as e:
                    raise ValueError(f"Failed to load override_neuron_config from file: {e}")
            elif not isinstance(self.override_neuron_config, dict):
                raise ValueError("override_neuron_config must be a dict or a valid file path")
        else:
            self.override_neuron_config = {}

        # handle quantization parameters
        if self.quant_dtype not in [None, "f8e4m3fn", "s8"]:
            raise ValueError("quant_dtype must be None, 'f8e4m3fn', or 's8'")
        if self.quant_dtype == "f8e4m3fn":
            with open(os.path.join(self.model_path, "modules_to_not_convert.json"), "r") as f:
                modules_config = json.load(f)

            # Set up neuron config for FP8 quantization
            self.override_neuron_config.update(
                {
                    "modules_to_not_convert": modules_config["model"]["modules_to_not_convert"],
                    "quantization_type": "per_channel_symmetric",
                    "quantization_dtype": "f8e4m3",
                }
            )
            # Set quant_dtype to None as FP8 quantization is handled internally using the override_neuron_config we just set up
            self.quant_dtype = None
        elif self.quant_dtype == "s8":
            self.quantization_param_path = os.path.join(self.model_path, "model_quant.pt")
        else:
            self.quant_dtype = None
