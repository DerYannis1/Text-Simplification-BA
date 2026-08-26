import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

HYPERPARAMETER_TUNING_DIR = "./mt5_hyperparameter_tuning"
TRAIN_SCRIPT = "./train_mt5.py"
TRAIN_FILE = "./dataset_mt5/train.jsonl"
VAL_FILE = "./dataset_mt5/val.jsonl"

HYPERPARAMETER_TUNING_DIR = [
    {"learning_rate": 3e-4, "num_train_epochs": 20, "label_smoothing_factor": 0.1},
    {"learning_rate": 1e-4, "num_train_epochs": 20, "label_smoothing_factor": 0.1},
    {"learning_rate": 3e-4, "num_train_epochs": 20, "label_smoothing_factor": 0.0},
    {"learning_rate": 3e-4, "num_train_epochs": 30, "label_smoothing_factor": 0.1},
    {"learning_rate": 3e-4, "num_train_epochs": 20, "optimizer": "adafactor"},
    {"learning_rate": 3e-4, "num_train_epochs": 20, "label_smoothing_factor": 0.1,
     "exclude_low_grounding": True},
]


def run_name(index, config):
    parts = [f"run{index:02d}"]
    for key, value in config.items():
        short_key = "".join(word[0] for word in key.split("_"))
        parts.append(f"{short_key}{value}")
    return "_".join(parts)


def build_command(config, output_directory):
    command = [
        sys.executable, TRAIN_SCRIPT,
        "--output-dir", str(output_directory),
        "--train-file", TRAIN_FILE,
        "--val-file", VAL_FILE,
    ]
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command += [flag, str(value)]
    return command


def run_single(config, name, dry_run):
    output_directory = Path(HYPERPARAMETER_TUNING_DIR) / name
    log_path = Path(HYPERPARAMETER_TUNING_DIR) / f"{name}.log"

    if (output_directory / "run_summary.json").exists():
        print(f"skippe [{name}]")
        return True

    command = build_command(config, output_directory)

    if dry_run:
        return True

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[{name}] failte nach {elapsed:.0f}s")
        return False

    print(f"[{name}] dauerte {elapsed:.0f}s")
    return True


def collect_results():
    rows = []
    for summary_path in sorted(Path(HYPERPARAMETER_TUNING_DIR).glob("*/run_summary.json")):
        with open(summary_path, encoding="utf-8") as file:
            summary = json.load(file)

        row = {"run_directory": summary_path.parent.name}
        row.update({k: v for k, v in summary["final_metrics"].items() if k.startswith("eval_")})
        for level, metrics in summary.get("per_level_metrics", {}).items():
            row[f"{level}_rougeL"] = metrics.get("rougeL")
        for key in ("learning_rate", "num_train_epochs", "label_smoothing_factor", "optimizer",
                    "exclude_low_grounding"):
            row[key] = summary["args"].get(key)
        rows.append(row)
    return rows


def write_comparison(rows):
    if not rows:
        print("keine results")
        return

    rows.sort(key=lambda row: row.get("eval_rougeL") or -1, reverse=True)

    csv_path = Path(HYPERPARAMETER_TUNING_DIR) / "hyperparameter_comparison.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done: {csv_path}")

    for row in rows:
        print(f"  {row['run_directory']:<40} rougeL={row.get('eval_rougeL', -1):.4f}")

    print(f"Bestes Modell: {rows[0]['run_directory']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_args()
    Path(HYPERPARAMETER_TUNING_DIR).mkdir(parents=True, exist_ok=True)

    for index, config in enumerate(HYPERPARAMETER_CONFIGS):
        run_single(config, run_name(index, config), arguments.dry_run)

    if not arguments.dry_run:
        rows = collect_results()
        write_comparison(rows)


if __name__ == "__main__":
    main()