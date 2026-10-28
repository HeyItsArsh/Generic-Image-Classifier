"""Train the reproducible transfer-learning candidate; installation is separate."""
import runpy
from pathlib import Path
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "upgrade" / "train_upgrade.py"), run_name="__main__")
