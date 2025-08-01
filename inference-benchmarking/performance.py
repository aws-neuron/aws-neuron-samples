import argparse
import json
import logging
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional

from clients.llm_perf import LLMPerfClient
from server.vllm import VLLMServer
from server_config import ServerConfig
from utils.artifacts import ArtifactManager

logger = logging.getLogger("PerformanceTesting")
logger.setLevel(logging.INFO)


def _get_performance_client(client: str, client_type: str = "llm_perf_github_patched"):
    """Get perf client based on type"""
    clients = {
        "llm_perf": LLMPerfClient(client_type=client_type),
        # change client type if needed
    }
    if client not in clients:
        raise ValueError(f"Unknown client type: {client}")
    return clients[client]


@dataclass
class PerformanceScenario:
    client: str
    max_concurrent_requests: int
    n_batches: int
    client_type: str = "llm_perf_github_patched"
    timeout: int = 3600
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    stddev_input_tokens: int = 0
    stddev_output_tokens: int = 0
    client_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        if self.n_batches < 1:
            raise ValueError("n_batches must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        if self.input_size is not None and self.input_size <= 0:
            raise ValueError("input_size must be positive if specified")
        if self.output_size is not None and self.output_size <= 0:
            raise ValueError("output_size must be positive if specified")
        if self.stddev_input_tokens < 0:
            raise ValueError("stddev_input_tokens must be non-negative")
        if self.stddev_output_tokens < 0:
            raise ValueError("stddev_output_tokens must be non-negative")


def run_perf_test(server_config: ServerConfig, named_scenarios: Dict[str, PerformanceScenario]):
    """
    Runs performance tests based on provided server configuration and scenarios.

    Args:
        server_config (ServerConfig): Configuration for the server
        named_scenarios (Dict[str, PerformanceScenario]): Dictionary mapping scenario names to their
                                                          corresponding PerformanceScenario objects

    Returns:
        Dict[str, Dict]: A nested dictionary structure where:
        - 'model_name': Name of the model being tested
        - 'scenarios': Dictionary containing test results organized by scenario, with structure:
            {
                scenario_name: {
                    'results': {
                        metric_name: score_value,
                        ...
                    },
                    'results_file': path_to_results_file
                }
            }
    """
    results_dict = {}
    results_dict["model_name"] = server_config.name
    results_dict["scenarios"] = defaultdict(dict)
    try:
        # download the artifacts
        artifact_manager = ArtifactManager()
        logger.info("Downloading model artifacts...")
        artifact_manager.download_model_artifacts(asdict(server_config))

        # start the server
        logger.info("Initializing VLLM server...")
        print(asdict(server_config))
        server = VLLMServer(**asdict(server_config))
        server_port, _, health = server.start()
        if not health:
            raise RuntimeError("Server failed to start")

        # client setup
        clients_cache = {}  # Cache clients by type
        for name, scenario in named_scenarios.items():
            if scenario.client not in clients_cache:
                clients_cache[scenario.client] = _get_performance_client(
                    scenario.client, scenario.client_type
                )
                clients_cache[scenario.client].setup()
            logger.info(f"Running performance scenario: {name}")

            client = clients_cache[scenario.client]
            results_dir = f"results/performance/{name}/"
            results, results_file = client.evaluate(
                model_path=server_config.model_path,
                server_port=server_port,
                results_dir=results_dir,
                max_concurrent_requests=scenario.max_concurrent_requests,
                input_size=scenario.input_size,
                output_size=scenario.output_size,
                n_batches=scenario.n_batches,
                timeout=scenario.timeout,
                **(scenario.client_params or {}),
            )
            results_dict["scenarios"][name] = {
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

        results_summary = run_perf_test(
            server_config=server_config, named_scenarios=test_config.performance
        )
        # Output results summary
        print("Evaluation completed successfully!")
        if results_summary:
            print("\nResults Summary:")
            for scenario_name, results_dict in results_summary["scenarios"].items():
                print(f"\n{scenario_name.capitalize()}:")
                results = results_dict["results"]
                results_file = results_dict["results_file"]
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
