import glob
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def safe_round(value):
    try:
        return round(float(value) * 100, 5)
    except (ValueError, TypeError):
        return value


class LMEvalClient:
    """LM-Eval accuracy evaluation client"""

    def __init__(self):
        self.scripts_dir = Path(__file__).parent / "scripts"

    def setup(self) -> None:
        """Setup LM-Eval environment"""
        setup_script = self.scripts_dir / "setup_lm_eval.sh"
        result = subprocess.run(["/bin/bash", str(setup_script)], check=True)
        if result.returncode != 0:
            raise RuntimeError("Failed to setup LM-Eval client")

        # Copy gated datasets from s3 to ~/.cache/huggingface/datasets/
        os.system(
            f"aws s3 sync --region=us-west-2 s3://kaena-nn-models/bat-accuracy-datasets/Idavidrein___gpqa/ ~/.cache/huggingface/datasets/Idavidrein___gpqa/ --only-show-errors"
        )

    def evaluate(
        self,
        model_path: str,
        server_port: int,
        task_name: str,
        results_dir: str,
        model_name: str = None,  # Optional model identifier
        max_concurrent_requests: int = 1,
        timeout: int = 3600,
        limit: int = 200,
        use_chat: bool = True,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Run LM-Eval evaluation

        Args:
            model_path: Path to model
            server_port: Port where model server is running
            task_name: Name of the task to evaluate
            results_dir: Directory to store results
            model_name: Optional model identifier (defaults to model_path)
            max_concurrent_requests: Number of concurrent requests
            timeout: Timeout in seconds
            limit: Number of examples to evaluate
            use_chat: Whether to use chat completions API

        Returns:
            Tuple of Dict containing evaluation results, and str result file path
        """
        run_script = self.scripts_dir / "run_lm_eval.sh"
        cmd = [
            "/bin/bash",
            str(run_script),
            model_name or model_path,  # Use model_name if provided, else model_path
            model_path,
            str(max_concurrent_requests),
            str(server_port),
            task_name,
            results_dir,
            str(timeout),
            str(limit),
            str(use_chat).lower(),  # Convert to lowercase string for shell script
            "2>&1 | tee -a client_out.txt",
        ]

        try:
            result = subprocess.run(
                " ".join(cmd), timeout=timeout, stderr=subprocess.STDOUT, text=True, shell=True
            )
            if result.returncode != 0:
                raise RuntimeError("LM-Eval evaluation failed")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Evaluation timed out after {timeout} seconds")

        # Process and return the results
        try:
            results_file = self.get_latest_results_file(results_dir)
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    results = json.load(f)
                return self._process_results(results, task_name=task_name), results_file
            else:
                raise FileNotFoundError(f"Results file not found: {results_file}")

        except FileNotFoundError:
            raise RuntimeError(f"Results file not found: {results_file}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse results file: {results_file}")

    def _process_results(self, results, task_name: str) -> Dict[str, Any]:
        """
        Process LMEval evaluation results
        """

        """Extract relevant metrics based on task type"""
        metrics_map = {}
        for subject_name, subject_result in results["results"].items():
            if "exact_match,strict-match" in subject_result:
                metrics_map[subject_name] = {
                    "score": safe_round(subject_result["exact_match,flexible-extract"]),
                    "exact_match,strict-match": safe_round(
                        subject_result["exact_match,strict-match"]
                    ),
                    "exact_match_stderr,strict-match": safe_round(
                        subject_result["exact_match_stderr,strict-match"]
                    ),
                    "exact_match,flexible-extract": safe_round(
                        subject_result["exact_match,flexible-extract"]
                    ),
                    "exact_match_stderr,flexible-extract": safe_round(
                        subject_result["exact_match_stderr,flexible-extract"]
                    ),
                }
            elif "prompt_level_strict_acc,none" in subject_result:
                metrics_map[subject_name] = {
                    "score": safe_round(
                        (
                            subject_result["prompt_level_strict_acc,none"]
                            + subject_result["inst_level_strict_acc,none"]
                        )
                        / 2
                    ),
                    "prompt_level_strict_acc,none": safe_round(
                        subject_result["prompt_level_strict_acc,none"]
                    ),
                    "prompt_level_strict_acc_stderr,none": safe_round(
                        subject_result["prompt_level_strict_acc_stderr,none"]
                    ),
                    "inst_level_strict_acc,none": safe_round(
                        subject_result["inst_level_strict_acc,none"]
                    ),
                    "prompt_level_loose_acc,none": safe_round(
                        subject_result["prompt_level_loose_acc,none"]
                    ),
                    "prompt_level_loose_acc_stderr,none": safe_round(
                        subject_result["prompt_level_loose_acc_stderr,none"]
                    ),
                    "inst_level_loose_acc,none": safe_round(
                        subject_result["inst_level_loose_acc,none"]
                    ),
                }
            elif "exact_match,none" in subject_result:
                metrics_map[subject_name] = {
                    "score": safe_round(subject_result["exact_match,none"]),
                    "exact_match,none": safe_round(subject_result["exact_match,none"]),
                    "exact_match_stderr,none": safe_round(
                        subject_result["exact_match_stderr,none"]
                    ),
                }
            elif "pass_at_1,none" in subject_result:
                metrics_map[subject_name] = {
                    "score": safe_round(subject_result["pass_at_1,none"]),
                    "pass_at_1,none": safe_round(subject_result["pass_at_1,none"]),
                    "pass_at_1_stderr,none": safe_round(subject_result["pass_at_1_stderr,none"]),
                }
            elif "exact_match,get-answer" in subject_result:
                metrics_map[subject_name] = {
                    "score": safe_round(subject_result["exact_match,get-answer"]),
                    "exact_match,get-answer": safe_round(subject_result["exact_match,get-answer"]),
                    "exact_match_stderr,get-answer": safe_round(
                        subject_result["exact_match_stderr,get-answer"]
                    ),
                }
            elif "exact_match,custom-extract" in subject_result:
                metrics_map[subject_name] = {
                    "score": safe_round(subject_result["exact_match,custom-extract"]),
                    "exact_match,custom-extract": safe_round(
                        subject_result["exact_match,custom-extract"]
                    ),
                    "exact_match_stderr,custom-extract": safe_round(
                        subject_result["exact_match_stderr,custom-extract"]
                    ),
                }
        return metrics_map

    def get_latest_results_file(self, directory):
        # Use glob to find all files matching the pattern in the directory and subdirectories
        pattern = os.path.join(directory, "**", "results_*.json")
        result_files = glob.glob(pattern, recursive=True)

        if not result_files:
            return None  # No matching files found

        # Function to extract datetime from filename
        def extract_datetime(filename):
            # Extract the datetime string from the filename
            date_str = os.path.basename(filename).split("results_")[1].split(".json")[0]
            # Parse the datetime string
            return datetime.strptime(date_str, "%Y-%m-%dT%H-%M-%S.%f")

        # Sort the files by the extracted datetime in descending order
        latest_file = max(result_files, key=extract_datetime)

        return latest_file
