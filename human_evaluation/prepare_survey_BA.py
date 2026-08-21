# File to run pipeline, for the examples in "./human_evalutation/source_text.jsonl"

import json
from pathlib import Path
from datetime import datetime
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pipeline import SimplificationPipeline

SOURCE_TEXT_PATH = "./human_evaluation/source_text.jsonl"
OUTPUT_DIR = "./human_evaluation"


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    output_dir = Path(OUTPUT_DIR)
    results_path = output_dir / "results_BA.jsonl"

    pipeline = SimplificationPipeline()

    source_texts = load_jsonl(SOURCE_TEXT_PATH)
    source_texts = source_texts[14:]

    for text in source_texts:
        source_text = text["text"]
        append_jsonl(results_path, source_text)
        result = pipeline.run(source_text, age_group="1")
        append_jsonl(results_path, result)
        print(f"done 1. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = pipeline.run(source_text, age_group="2")
        append_jsonl(results_path, result)
        print(f"done 2. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = pipeline.run(source_text, age_group="3")
        append_jsonl(results_path, result)
        print(f"done 3. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
 
if __name__ == "__main__":
    main()
