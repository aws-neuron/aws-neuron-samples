import argparse
import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional

from clients.lm_eval import LMEvalClient
from clients.long_bench import LongBenchClient
from server.vllm import VLLMServer
from server_config import ServerConfig
from utils.artifacts import ArtifactManager

logger = logging.getLogger("AccuracyTesting")
logger.setLevel(logging.INFO)

ACCURACY_CLIENTS_DATASETS = {
    "lm_eval": [
        "gsm8k_cot",
        "mmlu_flan_n_shot_generative",
        "leaderboard_ifeval",
        "leaderboard_math_hard",
        "bbh_cot_fewshot",
        "mmlu_pro",
        "gpqa_main_cot_n_shot",
        "mbpp",
    ],
    "longbench": ["single_document_qa", "multi_document_qa", "long_dialogue"],
}


def _get_accuracy_client(client: str):
    """Get evaluation client based on type"""
    clients = {
        "lm_eval": LMEvalClient(),
        "longbench": LongBenchClient(),
        # Add other accuracy client types here
    }

    if client not in clients:
        raise ValueError(f"Unknown client type: {client}")

    return clients[client]


@dataclass
class AccuracyScenario:
    client: str
    datasets: List[str]
    max_concurrent_requests: int = 1
    timeout: int = 3600
    client_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.client not in ACCURACY_CLIENTS_DATASETS.keys():
            raise ValueError(
                f"Unsupported accuracy client: {self.client}. "
                f"Must be one of {ACCURACY_CLIENTS_DATASETS.keys()}"
            )

        if not self.datasets:
            raise ValueError("At least one dataset must be specified")

        # Validate datasets for specific client
        valid_datasets = ACCURACY_CLIENTS_DATASETS.get(self.client, [])
        invalid_datasets = []
        for d in self.datasets:
            if not any(
                [d.startswith(v) for v in valid_datasets]
            ):  # all subsets of a task are valid
                invalid_datasets.append(d)

        if invalid_datasets:
            raise ValueError(
                f"Invalid datasets for client {self.client}: {invalid_datasets}. "
                f"Must be from {valid_datasets}"
            )

        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        # Client-specific parameter validation
        if self.client == "lm_eval":
            if "limit" in self.client_params and self.client_params["limit"] <= 0:
                raise ValueError("limit must be positive for lm_eval")
        elif self.client == "longbench":
            if "max_length" in self.client_params and self.client_params["max_length"] <= 0:
                raise ValueError("max_length must be positive for longbench")
            else:
                config_dir = Path(self.client_params["config_dir"])
                if config_dir.exists():
                    logging.info(f"\nDirectory exists: {config_dir}")
                else:
                    config_dir.mkdir(parents=True, exist_ok=True)
                    logging.info(f"\nDirectory created: {config_dir}")


def run_accuracy_test(server_config: ServerConfig, named_scenarios: Dict[str, AccuracyScenario]):
    """
    Runs accuracy tests based on provided server configuration and scenarios.

    Args:
        server_config (ServerConfig): Configuration for the server
        named_scenarios (Dict[str, AccuracyScenario]): Dictionary mapping scenario names to their corresponding AccuracyScenario objects

    Returns:
        Dict[str, Union[str, Dict]]: A nested dictionary structure where:
        - 'model_name': Name of the model being tested
        - 'scenarios': Dictionary containing test results organized by scenario, with structure:
            {
                scenario_name: {
                    dataset_name: {
                        'results': {
                            metric_name: score_value,
                            ...
                        },
                        'results_file': path_to_results_file
                    }
                }
            }

        Example:
        {
            'model_name': 'llama-3.1-70b-instruct',
            'scenarios': {
                'limited-gpu-tests': {
                    'gsm8k_cot': {
                        'results': {
                            'score': 97.5,
                            'exact_match,strict-match': 93.0,
                            ...
                        },
                        'results_file': 'path/to/results.json'
                    }
                }
            }
        }
    """
    results_dict = {}
    results_dict["model_name"] = server_config.name
    results_dict["scenarios"] = defaultdict(dict)
    tp_degree = server_config.tp_degree
    try:

        artifact_manager = ArtifactManager()
        logger.info("Downloading model artifacts...")
        artifact_manager.download_model_artifacts(asdict(server_config))

        logger.info("Initializing server...")
        config_dict = asdict(server_config)
        print(config_dict)

        server = VLLMServer(**config_dict)
        server_port, _, health = server.start()
        if not health:
            raise RuntimeError("Server failed to start")

        clients_cache = {}  # Cache clients by type
        for name, scenario in named_scenarios.items():
            if scenario.client not in clients_cache:
                clients_cache[scenario.client] = _get_accuracy_client(scenario.client)
                clients_cache[scenario.client].setup()
            logger.info(f"Running accuracy scenario: {name}")
            client = clients_cache[scenario.client]
            for dataset in scenario.datasets:
                results_dir = f"results/accuracy/{name}/{dataset}/"
                results, results_file = client.evaluate(
                    model_path=server_config.model_path,
                    server_port=server_port,
                    results_dir=results_dir,
                    task_name=dataset,
                    max_concurrent_requests=scenario.max_concurrent_requests,
                    timeout=scenario.timeout,
                    **(scenario.client_params or {}),
                )
                results_dict["scenarios"][name][dataset] = {
                    "results": results,
                    "results_file": results_file,
                }
    finally:
        # Cleanup
        server.cleanup()

    with open("results_summary.json", "w") as json_file:
        json.dump(results_dict, json_file, indent=4)
    return results_dict


def main(config_path):
    from utils.parser import ConfigParser

    try:
        # Parse configuration
        server_config, test_config = ConfigParser.parse_config(config_path)

        results_summary = run_accuracy_test(
            server_config=server_config, named_scenarios=test_config.accuracy
        )
        # Output results summary
        print("Evaluation completed successfully!")
        if results_summary:
            print("\nResults Summary:")
            for scenario_name, dataset_results_dict in results_summary["scenarios"].items():
                print(f"\n{scenario_name.capitalize()}:")
                for dataset_name, results_pair in dataset_results_dict.items():
                    results, results_file = results_pair["results"], results_pair["results_file"]
                    print(f"    Saved at  {results_file}:")
                    print(f"    Metrics:")
                    pprint(results)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Script with YAML config file")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        exit(1)
    main(config_path)
