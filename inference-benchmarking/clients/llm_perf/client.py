import json
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


class LLMPerfClient:
    """LLM Performance evaluation client"""

    SUPPORTED_CLIENTS = ["llm_perf", "llm_perf_github_patched"]

    def __init__(self, client_type: str = "llm_perf_github_patched"):
        if client_type not in self.SUPPORTED_CLIENTS:
            raise ValueError(
                f"Unsupported client type: {client_type}. Must be one of {self.SUPPORTED_CLIENTS}"
            )
        self.client_type = client_type
        self.scripts_dir = Path(__file__).parent / "scripts"

        if client_type == "llm_perf_github_patched":
            self.llmperf_dir = Path.home() / "llmperfGithubPatched"
        else:
            self.llmperf_dir = self.scripts_dir / "llmperf"

    def setup(self) -> None:
        """Setup LLM-Perf environment"""
        setup_script = self.scripts_dir / "setup_llm_perf.sh"

        # Ensure target directory exists and is accessible
        os.makedirs(self.llmperf_dir, exist_ok=True)

        result = subprocess.run(["/bin/bash", str(setup_script), self.client_type], check=True)
        if result.returncode != 0:
            raise RuntimeError("Failed to setup LLM-Perf client")

    def evaluate(
        self,
        model_path: str,
        server_port: int,
        max_concurrent_requests: int,
        input_size: int,
        output_size: int,
        n_batches: int,
        results_dir: str,
        stddev_input_tokens: int = 0,
        stddev_output_tokens: int = 0,
        tokenizer: Optional[str] = None,
        timeout: int = 3600,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run LLM-Perf benchmark"""
        run_script = self.scripts_dir / "run_llm_perf.sh"

        if not run_script.exists():
            raise FileNotFoundError(f"Script not found: {run_script}")

        results_dir_abs = os.path.abspath(results_dir)
        cmd = [
            "/bin/bash",
            "-x",  # Add -x for debug output
            str(run_script),
            model_path,
            str(max_concurrent_requests),
            str(input_size),
            str(stddev_input_tokens),
            str(output_size),
            str(stddev_output_tokens),
            results_dir_abs,
            str(n_batches),
            str(server_port),
        ]

        cmd.extend(["--client-type", self.client_type])

        if tokenizer:
            cmd.extend(["--tokenizer", tokenizer])

        print("\n" + "=" * 80)
        print("Executing LLM-Perf benchmark")
        print(f"Command: {' '.join(cmd)}")
        print(f"Working directory: {os.getcwd()}")
        print("=" * 80 + "\n")

        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={
                    **os.environ.copy(),
                    "OPENAI_API_KEY": "EMPTY",
                    "OPENAI_API_BASE": f"http://localhost:{server_port}/v1",
                    "LLMPERF_INSTALL_DIR": str(self.llmperf_dir),
                },
                cwd=os.path.dirname(run_script),
            )

            print("\n" + "=" * 80)
            print("Command completed successfully!")
            print("\nCommand output:")
            print(result.stdout)
            print("=" * 80 + "\n")

            results_data = self.process_results(
                results_dir=results_dir,
                input_size=input_size,
                output_size=output_size,
                model_path=model_path,
                stddev_input_tokens=stddev_input_tokens,
                stddev_output_tokens=stddev_output_tokens,
            )

            return results_data, results_data.get("results_file", "unknown_file_path")

        except subprocess.CalledProcessError as e:
            print("\n" + "=" * 80)
            print("Command execution failed!")
            print(f"Return code: {e.returncode}")
            print("\nCommand output:")
            print(e.stdout if e.stdout else "No stdout output")
            print("\nCommand error:")
            print(e.stderr if e.stderr else "No stderr output")
            print("=" * 80 + "\n")

            if os.path.exists(results_dir):
                print("\nContents of results directory:")
                for file in os.listdir(results_dir):
                    print(f"- {file}")
                print()

            raise RuntimeError(
                f"LLM-Perf benchmark failed with return code {e.returncode}. "
                f"Check the output above for details. "
                f"Working directory: {os.getcwd()}"
            )
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"Unexpected error: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            print("=" * 80 + "\n")
            raise

    def process_results(
        self,
        results_dir: str,
        input_size: int,
        output_size: int,
        model_path: str,
        stddev_input_tokens: int = 0,
        stddev_output_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Process benchmark results"""
        possible_locations = [
            results_dir,
            os.path.join(os.path.expanduser("~"), results_dir),
            os.path.join(os.path.expanduser("~"), "results/performance/mytest/"),
        ]

        base_name = model_path.replace(".", "-").replace("/", "-")
        filename_pattern = f"{base_name}_{input_size}_{output_size}"
        if stddev_input_tokens > 0 or stddev_output_tokens > 0:
            filename_pattern += f"_stddev_{stddev_input_tokens}_{stddev_output_tokens}"

        results_file = None
        for location in possible_locations:
            print(f"Checking location: {location}")
            summary_path = os.path.join(location, f"{filename_pattern}_summary.json")

            if os.path.exists(summary_path):
                results_file = summary_path
                print(f"Found summary file: {results_file}")
                break

            if os.path.exists(location):
                print(f"Contents of {location}:")
                for file in os.listdir(location):
                    print(f"  - {file}")
                    if file.endswith("_summary.json"):
                        results_file = os.path.join(location, file)
                        print(f"Found summary file: {results_file}")
                        break

        if results_file is None:
            for location in possible_locations:
                individual_path = os.path.join(
                    location, f"{filename_pattern}_individual_responses.json"
                )
                if os.path.exists(individual_path):
                    print(f"Found individual responses file: {individual_path}")
                    print("Summary file might be in the same directory")

                    # Look for any summary file in this directory
                    dir_path = os.path.dirname(individual_path)
                    for file in os.listdir(dir_path):
                        if file.endswith("_summary.json"):
                            results_file = os.path.join(dir_path, file)
                            print(f"Found summary file: {results_file}")
                            break

        if results_file is None:
            raise FileNotFoundError(f"Results file not found in any of the expected locations")

        print(f"Processing results from: {results_file}")
        with open(results_file, "r") as f:
            metrics = json.load(f)
            print(metrics)

        return {
            "e2e_model": {
                "latency_ms_p50": metrics["results_end_to_end_latency_s_quantiles_p50"] * 1000,
                "latency_ms_p90": metrics["results_end_to_end_latency_s_quantiles_p90"] * 1000,
                "latency_ms_p95": metrics["results_end_to_end_latency_s_quantiles_p95"] * 1000,
                "latency_ms_p99": metrics["results_end_to_end_latency_s_quantiles_p99"] * 1000,
                "latency_ms_p100": metrics["results_end_to_end_latency_s_max"] * 1000,
                "throughput": (
                    metrics["results_number_input_tokens_mean"]
                    + metrics["results_number_output_tokens_mean"]
                )
                / metrics["results_end_to_end_latency_s_mean"],
            },
            "context_encoding_model": {
                "latency_ms_p50": metrics["results_ttft_s_quantiles_p50"] * 1000,
                "latency_ms_p90": metrics["results_ttft_s_quantiles_p90"] * 1000,
                "latency_ms_p95": metrics["results_ttft_s_quantiles_p95"] * 1000,
                "latency_ms_p99": metrics["results_ttft_s_quantiles_p99"] * 1000,
                "latency_ms_p100": metrics["results_ttft_s_max"] * 1000,
                "throughput": metrics["results_number_input_tokens_mean"]
                / metrics["results_ttft_s_mean"],
            },
            "token_generation_model": {
                "latency_ms_p50": metrics["results_inter_token_latency_s_quantiles_p50"] * 1000,
                "latency_ms_p90": metrics["results_inter_token_latency_s_quantiles_p90"] * 1000,
                "latency_ms_p95": metrics["results_inter_token_latency_s_quantiles_p95"] * 1000,
                "latency_ms_p99": metrics["results_inter_token_latency_s_quantiles_p99"] * 1000,
                "latency_ms_p100": metrics["results_inter_token_latency_s_max"] * 1000,
                "per_req_throughput": metrics["results_request_output_throughput_token_per_s_mean"],
                "throughput": metrics["results_mean_output_throughput_token_per_s"],
            },
            "raw_metrics": metrics,
            "results_file": results_file,
        }
