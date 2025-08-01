import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.s3 import download_from_s3


class ArtifactManager:
    """Manages model artifacts and test artifacts"""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path("artifacts")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_model_artifacts(self, model_config: Dict[str, Any]) -> None:
        """Download model and related artifacts"""
        print(model_config)
        # Download main model
        if model_config.get("model_s3_path"):
            download_from_s3(model_config["model_s3_path"], model_config["model_path"])

        # Download draft model if specified
        if model_config.get("draft_model_s3_path"):
            download_from_s3(model_config["draft_model_s3_path"], model_config["draft_model_path"])

        # Download sharded weights if specified
        if model_config.get("sharded_weights_s3_path"):
            download_from_s3(
                model_config["sharded_weights_s3_path"], model_config["sharded_weights_path"]
            )

    def save_artifacts(self, artifacts: Dict[str, Path], destination: str) -> None:
        """Save artifacts to specified destination"""
        for name, path in artifacts.items():
            if path.is_file():
                shutil.copy2(path, self.base_dir / destination / name)
            elif path.is_dir():
                shutil.copytree(path, self.base_dir / destination / name)

    def upload_to_s3(self, local_path: Path, s3_path: str, recursive: bool = False) -> bool:
        """Upload artifacts to S3"""
        cmd = ["aws", "s3"]
        cmd.extend(["sync" if recursive else "cp"])
        cmd.extend([str(local_path), s3_path])

        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to upload to S3: {e}")
            return False

    def cleanup(self, paths: List[Path]) -> None:
        """Cleanup artifact paths"""
        for path in paths:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
