import argparse
import logging
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
        "mmlu_flan_n_shot_generative_computer_security",
        "mmlu_flan_n_shot_generative_logical_fallacies",
        "mmlu_flan_n_shot_generative_nutrition",
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
        invalid_datasets = [d for d in self.datasets if d not in valid_datasets]
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


def run_accuracy_test(server_config: ServerConfig, named_scenarios: Dict[str, AccuracyScenario]):
    name = server_config.name
    tp_degree = server_config.tp_degree
    try:
        artifact_manager = ArtifactManager()
        logger.info("Downloading model artifacts...")
        artifact_manager.download_model_artifacts(asdict(server_config))

        logger.info("Initializing server...")
        print(asdict(server_config))
        server = VLLMServer(**asdict(server_config))
        server_port, _, health = server.start()
        if not health:
            raise RuntimeError("Server failed to start")

        results_dict = {}
        clients_cache = {}  # Cache clients by type
        for name, scenario in named_scenarios.items():
            if scenario.client not in clients_cache:
                clients_cache[scenario.client] = _get_accuracy_client(scenario.client)
                clients_cache[scenario.client].setup()
            logger.info(f"Running accuracy scenario: {name}")
            client = clients_cache[scenario.client]
            for dataset in scenario.datasets:
                results_dir = f"results/accuracy/{name}/{dataset}/"
                results = client.evaluate(
                    model_path=server_config.model_path,
                    server_port=server_port,
                    results_dir=results_dir,
                    task_name=dataset,
                    max_concurrent_requests=scenario.max_concurrent_requests,
                    timeout=scenario.timeout,
                    **(scenario.client_params or {}),
                )
                results_dict[f"accuracy_{name}_{dataset}"] = results
    finally:
        # Cleanup
        server.cleanup()

    return results_dict


def main(config_path):
    from utils.parser import ConfigParser

    try:
        # Parse configuration
        server_config, test_config = ConfigParser.parse_config(config_path)

        results_dict = run_accuracy_test(
            server_config=server_config, named_scenarios=test_config.accuracy
        )
        # Output results summary
        print("Evaluation completed successfully!")
        if results_dict:
            print("\nResults Summary:")
            for test_name, test_results in results_dict.items():
                print(f"\n{test_name.capitalize()}:")
                results, results_dir = test_results
                print(f"    Saved at  {results_dir}:")
                print(f"    Metrics: {results}")

    except Exception as e:
        print(f"Error: {str(e)}", err=True)
        print("\nFull traceback:", err=True)
        print(traceback.format_exc(), err=True)
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
