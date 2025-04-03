from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ServerConfig:
    name: str
    model_path: str
    model_s3_path: str
    max_seq_len: int
    context_encoding_len: int
    tp_degree: int
    n_vllm_threads: int
    server_port: int
    continuous_batch_size: int = 1

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
