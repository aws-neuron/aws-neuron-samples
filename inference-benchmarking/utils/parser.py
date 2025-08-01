from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

from accuracy import AccuracyScenario
from performance import PerformanceScenario
from server_config import ServerConfig


@dataclass
class TestConfig:
    accuracy: Dict[str, AccuracyScenario] = field(default_factory=dict)
    performance: Dict[str, PerformanceScenario] = field(default_factory=dict)
    upload_artifacts: bool = False

    def __post_init__(self):
        # Ensure at least one type of test is configured
        if not self.accuracy and not self.performance:
            raise ValueError("At least one test type (accuracy or performance) must be configured")


class ConfigParser:
    @staticmethod
    def parse_config(config_path: str) -> tuple[ServerConfig, TestConfig]:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Validation happens during dataclass instantiation
        server_config = ServerConfig(**config["server"])
        test_config = TestConfig(
            accuracy={
                name: AccuracyScenario(**scenario_config)
                for name, scenario_config in config["test"].get("accuracy", {}).items()
            },
            performance={
                name: PerformanceScenario(**scenario_config)
                for name, scenario_config in config["test"].get("performance", {}).items()
            },
        )

        return server_config, test_config
