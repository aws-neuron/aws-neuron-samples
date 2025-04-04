import glob
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
        ]

        try:
            result = subprocess.run(cmd, timeout=timeout, check=True)
            if result.returncode != 0:
                raise RuntimeError("LM-Eval evaluation failed")

            results_file = self.get_latest_results_file(results_dir)
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    results = json.load(f)
                return self._process_results(results, task_name=task_name), results_file
            else:
                raise FileNotFoundError(f"Results file not found: {results_file}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Evaluation timed out after {timeout} seconds")
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
            if "exact_match,strict-match" not in subject_result:
                continue
            metric_data = {
                "AccuracyExactMatchStrictMatch": round(
                    subject_result["exact_match,strict-match"] * 100, 5
                ),
                "AccuracyExactMatchStrictMatchStderr": round(
                    subject_result["exact_match_stderr,strict-match"] * 100, 5
                ),
                "AccuracyExactMatchFlexibleExtract": round(
                    subject_result["exact_match,flexible-extract"] * 100, 5
                ),
                "AccuracyExactMatchFlexibleExtractStderr": round(
                    subject_result["exact_match_stderr,flexible-extract"] * 100, 5
                ),
            }
            metrics_map[subject_name] = metric_data
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
