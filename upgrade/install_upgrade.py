"""Install an evaluated candidate and back up the active model/labels."""
from pathlib import Path
import json, shutil, datetime
ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "upgrade"

def main():
    for name in ("image_classifier.keras", "classes.json", "comparison.json"):
        if not (CANDIDATE / name).is_file():
            raise SystemExit(f"Missing candidate {name}. Run Main.py to train and evaluate first.")
    report = json.loads((CANDIDATE / "comparison.json").read_text())
    if report["holdout"]["new"]["balanced_accuracy"] <= report["holdout"]["old"]["balanced_accuracy"]:
        raise SystemExit("Candidate did not improve holdout balanced accuracy; not installing.")
    labels = json.loads((CANDIDATE / "classes.json").read_text())
    if labels != json.loads((ROOT / "classes.json").read_text()):
        raise SystemExit("Candidate category order differs; review before installation.")
    backup = ROOT / "backups" / datetime.datetime.now().strftime("before_install_%Y%m%d_%H%M%S_%f")
    backup.mkdir(parents=True)
    for name in ("image_classifier.keras", "classes.json"):
        shutil.copy2(ROOT / name, backup / name)
    for name in ("image_classifier.keras", "classes.json"):
        shutil.copy2(CANDIDATE / name, ROOT / name)
    print("Installed candidate. Restart the image desk. Backup:", backup)

if __name__ == "__main__":
    main()
