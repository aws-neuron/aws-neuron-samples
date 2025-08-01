import errno
import json
import logging
import os
import signal
import socket
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import psutil
import requests

from utils.tee_output import TeeOutput

from utils import (
    check_server_terminated,
    download_from_s3,
    find_free_port,
    is_port_available,
    kill_process_and_children,
)

LARGE_LLM_LIST = ["dbrx", "llama-3.3-70B", "llama-3.1-405b"]


class VLLMServer:
    """Managing VLLM server deployment"""

    def __init__(
        self,
        name: str,  # Unique name identifier for the server instance
        model_path: str,  # Path to the model weights/config
        continuous_batch_size: int,  # Batch size for continuous batching
        ctx_output_lengths: Union[List[Tuple[int]], Tuple[int]] = (None, None),
        max_seq_len: int = None,  # Maximum sequence length for the model
        tp_degree: int = 24,  # Tensor parallelism degree
        n_vllm_threads=32,  # Number of threads for VLLM
        quant_dtype=None,  # Quantization data type if using quantization
        quantization_param_path=None,  # Path to quantized weights
        enabled_chunked_prefill=False,
        chunk_size=0,
        block_size=0,
        num_blocks_override=0,
        custom_chat_template_path=None,  # Path to custom chat template file
        server_port=8000,  # Port number for server
        draft_model_path=None,  # Path to draft model for speculative decoding
        spec_len=None,
        speculation_type=None,
        compiled_model_path=None,
        inference_demo_script=None,
        inference_demo_args=None,
        logical_neuron_cores=1,
        scratchpad_page_size=None,
        enable_scratchpad_single_core_debugging=False,
        override_neuron_config: dict = None,
        **kwargs,
    ):
        self.logger = logging.getLogger("VLLMServer")
        self.logger.setLevel(logging.INFO)
        self.inference_demo_script = None
        if inference_demo_script:
            if not compiled_model_path:
                raise ValueError(
                    f"compiled_model_path is required when inference_demo_script is used to instantiate vllm server"
                )
            self.inference_demo_script = inference_demo_script
        self.inference_demo_args = inference_demo_args
        self.name = name
        self.model_path = model_path
        self.cont_batch_size = continuous_batch_size
        self.tp_degree = tp_degree
        self.n_vllm_threads = n_vllm_threads
        self.quant_dtype = quant_dtype
        if self.quant_dtype:
            assert quantization_param_path is not None, "quantization_param_path is required"
            self.quantization_param_path = quantization_param_path
        self.enabled_chunked_prefill = enabled_chunked_prefill
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.num_blocks_override = num_blocks_override
        self.compiled_model_path = compiled_model_path
        self.custom_chat_template_path = custom_chat_template_path
        self.server_port = server_port
        self.draft_model_path = draft_model_path
        self.spec_len = spec_len
        self.speculation_type = speculation_type
        self.logical_neuron_cores = logical_neuron_cores
        self.scratchpad_page_size = scratchpad_page_size
        self.enable_scratchpad_single_core_debugging = enable_scratchpad_single_core_debugging
        self.process = None
        mean_input_tokens, mean_output_tokens = ctx_output_lengths

        if bool(max_seq_len) and ctx_output_lengths != (
            None,
            None,
        ):  # Ensures exactly one is provided
            raise ValueError(
                "Either max_seq_len or ctx_output_lengths should be passed to vllm server, but not both."
            )

        self.max_seq_len = (
            max_seq_len if max_seq_len else 2 * (mean_input_tokens + mean_output_tokens)
        )
        self.cores = f"0-{self.tp_degree - 1}"
        self.vllm_tokenizer = kwargs.get("vllm_tokenizer", None)

        self.override_neuron_config = {}
        if override_neuron_config is not None:
            self.override_neuron_config = override_neuron_config

        self.scripts_dir = Path(__file__).parent / "scripts"

    def run_inference_demo(self):
        """Run the inference_demo script to compile artifacts."""
        cmd = [
            "/bin/bash",
            self.inference_demo_script,
        ]

        if self.inference_demo_args:
            cmd += self.inference_demo_args.split(" ")

        try:
            result = subprocess.run(cmd, check=True)

            # check that compilation succeeded, inference_demo can sometimes report a zero error code despite failure
            assert os.path.exists(os.path.join(self.compiled_model_path, "model.pt"))

            print(f"Inference demo completed successfully. {result.returncode}")
        except subprocess.CalledProcessError as e:
            # Raise an error if the script fails
            print(f"Inference demo failed with exit code {e.returncode}")
            raise

    def start_vllm_server(self):
        try:
            # Avoid recompilation by setting this env variable
            if self.compiled_model_path:
                os.environ["NEURON_COMPILED_ARTIFACTS"] = self.compiled_model_path
            if self.vllm_tokenizer:
                os.environ["VLLM_TOKENIZER"] = self.vllm_tokenizer

            # If scratchpad page size is changed from default, set it here
            if self.scratchpad_page_size:
                os.environ["NEURON_SCRATCHPAD_PAGE_SIZE"] = str(self.scratchpad_page_size)
            if self.enable_scratchpad_single_core_debugging:
                os.environ["NEURON_RT_DBG_SCRATCHPAD_ON_SINGLE_CORE"] = "1"

            # Populate compiled artifacts using inference_demo()
            if hasattr(self, "inference_demo_script") and self.inference_demo_script:
                self.run_inference_demo()

            # Sometimes, OS takes a while to release the server port after a given test case finishes.
            # When running multiple configs consecutively this can cause EADDRINUSE error for some subsequent test cases.
            # Use next available port in such cases and return the port used
            port = self.server_port
            if not is_port_available(port):
                print(f"Port {port} is in use. Using a different port.")
                port = find_free_port(start_port=port)
                print(f"Found available {port} for server.")
            self.server_port = port

            vllm_start_script = self.scripts_dir / "start_server.sh"

            args = [
                "/bin/bash",
                str(vllm_start_script),
                self.model_path,
                str(port),
                self.cores,
                str(self.max_seq_len),
                str(self.cont_batch_size),
                str(self.tp_degree),
                str(self.n_vllm_threads),
            ]

            # Add optional arguments with parameter names
            if self.draft_model_path is not None:
                args.extend(["--speculative-model", self.draft_model_path])
            if self.spec_len is not None:
                args.extend(
                    [
                        "--num-speculative-tokens",
                        str(self.spec_len),
                    ]
                )
            if self.custom_chat_template_path is not None:
                args.extend(
                    [
                        "--chat-template",
                        (
                            str(Path(__file__).parent / "scripts" / "prompt-template.jinja")
                            if self.custom_chat_template_path == "default"
                            else self.custom_chat_template_path
                        ),
                    ]
                )
            if self.quant_dtype is not None:
                assert (
                    self.quantization_param_path is not None
                ), "quantization_param_path is required when quant_dtype is set"
                warnings.warn(
                    "quant_dtype and quantization_param_path passed directly to VllmServer, set neuron-specific flags in override_neuron_config instead",
                    category=UserWarning,
                )
                self.override_neuron_config.update(
                    {
                        "quantized": True,
                        "quantized_checkpoints_path": self.quantization_param_path,
                        "quantization_type": "per_channel_symmetric",
                    }
                )
            if self.speculation_type is not None:
                warnings.warn(
                    "speculation_type is passed directly to VllmServer, set neuron-specific flags in override_neuron_config instead",
                    category=UserWarning,
                )
                if self.speculation_type == "eagle":
                    self.override_neuron_config.update({"enable_eagle_speculation": True})
                else:
                    self.override_neuron_config.update({"enable_fused_speculation": True})
            if self.enabled_chunked_prefill:
                args.extend(["--enable-chunked-prefill", str(True)])
                args.extend(["--max-num-batched-tokens", str(self.chunk_size)])
                args.extend(["--block-size", str(self.block_size)])
                args.extend(["--num-gpu-blocks-override", str(self.num_blocks_override)])
            if self.logical_neuron_cores > 1:
                warnings.warn(
                    "logical_neuron_cores passed directly to VllmServer, set neuron-specific flags in override_neuron_config instead",
                    category=UserWarning,
                )
                self.override_neuron_config.update(
                    {"logical_neuron_cores": self.logical_neuron_cores}
                )
            if self.override_neuron_config:
                args.extend(
                    ["--override-neuron-config", f"'{json.dumps(self.override_neuron_config)}'"]
                )

            # Start server
            # NOTE: this step includes compilation if model is not compiled outside Vllm
            args.append("2>&1 | tee -a server_out.txt")
            print("command: ", " ".join(args))
            process = subprocess.Popen(
                " ".join(args), text=True, stderr=subprocess.STDOUT, shell=True
            )
            # Wait a bit for the server to start.
            # Sometimes server start can throw the following error if it wasn't terminated gracefully in a prior run. NRT (Neuron Runtime) can get confused in such cases and not have cores available
            # ERROR   NRT:nrt_allocate_neuron_cores Logical Neuron Core(s) not available - Requested:lnc0-lnc15 Available:0 Logical Core size:1
            time.sleep(60)
            if process.poll() is not None:  # Check process is alive
                raise RuntimeError(
                    f"subprocess return code: {process.returncode}"
                )  # failure reason will be printed on console before reaching here
        except Exception as e:
            print(f"Failed to start the subprocess: {e}")
            raise RuntimeError(f"Server instantiation failed due to {e}")

        health_check = self.check_health_endpoint(f"http://localhost:{port}/health")
        return port, process, health_check

    def check_health_endpoint(self, url, num_retries=None, delay=30):
        retries = num_retries if num_retries else self._get_num_retries_for_model()
        for i in range(retries):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print("Server is up! Health check passed.")
                    return True
            except requests.ConnectionError:
                print(
                    f"Attempt {i + 1}/{retries}: Server is not ready yet. Retrying in {delay} seconds..."
                )
            time.sleep(delay)

        print("Server did not respond within the retry limit.")
        return False

    def start(self):
        """
        Starts the VLLM server and performs health checks

        This function:
        1. Checks if a server is already running on the port and terminates it if needed
        2. Starts either a speculative or regular VLLM server based on configuration
        3. Performs health checks to ensure server started successfully

        Returns:
                (int, process, bool): Server port number, server process, and health check status

        Raises:
                RuntimeError: If unable to kill pre-running server
                ConnectionRefusedError: If server fails to start successfully
        """
        self.logger.info("Starting VLLM server...")
        health_check_url = f"http://localhost:{self.server_port}/health"
        self.logger.info(f"Checking server health at {health_check_url}")
        server_terminated = check_server_terminated(health_check_url, 1, 0)
        self.logger.info(f"Server terminated status: {server_terminated}")
        if server_terminated is not True:
            self.logger.warning("Server from previous test still up. Attempting to kill server")
            print("Server from previous test still up. Attempting to kill server")
            self.kill_children_of_process_on_port(self.server_port)
            server_terminated = check_server_terminated(health_check_url, 2, 10)
            self.logger.info(f"Server terminated after kill attempt: {server_terminated}")
            if server_terminated is not True:
                self.logger.error("Unable to kill pre-running server")
                raise RuntimeError("Unable to kill pre-running server. Skipping tests.")

        self.logger.info("Starting VLLM server process...")
        server_port, server_process, health_check = self.start_vllm_server()
        self.logger.info(f"Server process started. Port: {server_port}, Health: {health_check}")
        if health_check is False:
            self.logger.error("Server did not start successfully")
            raise ConnectionRefusedError("Server did not start successfully. Skipping tests.")

        return server_port, server_process, health_check

    def cleanup(self):
        self.kill_children_of_process_on_port(self.server_port)

    def kill_children_of_process_on_port(self, port):
        # Store unique pid for each matching connection in a set. Example of multiple conn object for the same pid below
        # sconn(fd=26, family=<AddressFamily.AF_INET: 2>, type=<SocketKind.SOCK_STREAM: 1>, laddr=addr(ip='0.0.0.0', port=8000), raddr=(), status='LISTEN', pid=403626)
        # sconn(fd=28, family=<AddressFamily.AF_INET6: 10>, type=<SocketKind.SOCK_STREAM: 1>, laddr=addr(ip='::', port=8000), raddr=(), status='LISTEN', pid=403626)
        pids = set()
        for connection in psutil.net_connections():
            if connection.laddr.port == port:
                print(f"Connection: {connection}")
                if connection.pid:  # to handle TIME_WAIT connections with pid=None
                    pids.add(connection.pid)

        for pid in pids:
            kill_process_and_children(pid)
        return None

    def _get_num_retries_for_model(self):
        """
        Estimate the number of retries when vllm loads a compiled NXDI model.

        Large model will take a longer time because it needs to load more weights to device.
        """
        model_name_or_path = self.name + self.model_path
        is_large_model = any([target in model_name_or_path for target in LARGE_LLM_LIST])

        if is_large_model:
            return 200
        else:
            return 120
