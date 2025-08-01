# aws_neuron_eval/utils/tee_output.py
import sys
from datetime import datetime
from pathlib import Path


class TeeOutput:
    """
    A file-like object that writes to both a file and stdout.
    """

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.file_path, "w", buffering=1)  # Line buffering
        self._stdout = sys.stdout
        # Write initial message to verify file writing
        self.write(f"=== Log started at {datetime.now()} ===\n")

    def write(self, data):
        try:
            self._file.write(data)
            self._file.flush()  # Force write to disk
            self._stdout.write(data)
            self._stdout.flush()
        except Exception as e:
            print(f"Error writing to log: {e}")

    def flush(self):
        try:
            self._file.flush()
            self._stdout.flush()
        except Exception as e:
            print(f"Error flushing: {e}")

    def fileno(self):
        return self._stdout.fileno()

    def close(self):
        if not self._file.closed:
            self.write(f"\n=== Log ended at {datetime.now()} ===\n")
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.write(f"\nError occurred: {exc_val}\n")
        self.close()

    @classmethod
    def create_with_timestamp(cls, base_path, prefix):
        log_file = create_log_with_timestamp(base_path, prefix)
        return cls(log_file)


def create_log_with_timestamp(base_path, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{prefix}_{timestamp}.log"
    print(f"Creating log file at: {log_file}")
    return log_file
