from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class EvalClient(ABC):
    """Base class for evaluation clients"""

    def __init__(self):
        self.scripts_dir = Path(__file__).parent

    @abstractmethod
    def setup(self) -> None:
        """Setup the client (install dependencies, etc.)"""
        pass

    @abstractmethod
    def run(self, server_port: int, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run evaluation and return standardized results

        Returns:
            Dict with standardized format:
            {
                "metrics": {
                    "metric_name": value,
                    ...
                },
                "metadata": {
                    "scenario": str,
                    "client": str,
                    "timestamp": str,
                    ...
                },
                "raw_results": Dict  # Original client output
            }
        """
        pass

    def _get_script_path(self, script_name: str) -> str:
        return str(self.scripts_dir / script_name)
