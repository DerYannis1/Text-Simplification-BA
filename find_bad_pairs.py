import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

SIMILARITY_THRESHOLD = 0.7
DATASET_PATH = "./dataset/test.jsonl"
OUTPUT_CSV_PATH = "./dataset"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

#load entries
def load_entries(file_path):
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            missing = {"id", "source", "target", "level"} - obj.keys()
            if missing:
                continue
            entries.append(obj)
    return entries

#compute source target similarity
def compute_source_target_similarity(entries, model):
    sources = [e["source"] for e in entries]
    targets = [e["target"] for e in entries]

    print(f"{len(sources)} sources found")
    print(f"{len(targets)} targets found")
    emb_sources = model.encode(sources, convert_to_tensor=True, show_progress_bar=True, batch_size=16)
    emb_targets = model.encode(targets, convert_to_tensor=True, show_progress_bar=True, batch_size=16)

    results = []
    for i, e in enumerate(entries):
        cosine_sim = util.cos_sim(emb_sources[i], emb_targets[i]).item()
        results.append({
            "id": e["id"],
            "level": e["level"],
            "comparison": "source_vs_target",
            "similarity": round(cosine_sim, 4)
        })
    return results

#compute cross level similarity
def compute_cross_level_similarity(entries, model):

    by_articles: dict[str, dict[str, str]] = defaultdict(dict)
    for e in entries:
        by_articles[e["id"]][e["level"]] = e["target"]

    pairs = [
        (id_, texts["Klexikon"], texts["MiniKlexikon"])
        for id_, texts in by_articles.items()
        if "Klexikon" in texts and "MiniKlexikon" in texts
    ]

    articles = [p[0] for p in pairs]
    klex_texts = [p[1] for p in pairs]
    mini_texts = [p[2] for p in pairs]

    print(f"{len(klex_texts)} klexikon entries found")
    print(f"{len(mini_texts)} miniklexikon entries found")
    emb_klex = model.encode(klex_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=16)
    emb_mini = model.encode(mini_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=16)

    results = []
    for i, e in enumerate(articles):
        cosine_sim = util.cos_sim(emb_klex[i], emb_mini[i]).item()
        results.append({
            "id": e["id"],
            "level": e["level"],
            "comparison": "source_vs_target",
            "similarity": round(cosine_sim, 4)
        })
    return results

#create outlier list
def catch_bad_pairs(results, similarity_treshold):
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in results:
        grouped[(r["level"], r["comparison"])].append(r["similarity"])

    thresholds = {}
    for key, values in grouped.items():
        sorted_values = sorted(values)
        idx = max(0, int(len(sorted_values) * similarity_treshold /100 ) - 1)
        thresholds[key] = sorted_values[idx]

    for r in results:
        key = (r["level"], r["comparison"])

    return results

#write to csv file
def write_to_csv(results, out_path):
    field_names = ["id", "level", "comparison", "similarity"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        # lowest similarity first
        for r in sorted(results, key=lambda x: x["similarity"]):
            writer.writerow(r)
    print("written to csv file")
#main 
def main():
    file_path = Path(DATASET_PATH)
    output_src_tar_path = Path(OUTPUT_CSV_PATH + "/src_tar.csv")
    output_cross_path = Path(OUTPUT_CSV_PATH + "/cross.csv")
    threshold = SIMILARITY_THRESHOLD

    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    entries = load_entries(file_path)
    print(f" {len(entries)} entries loaded")

    model = SentenceTransformer(MODEL_NAME)
    results = compute_source_target_similarity(entries, model)
    
    write_to_csv(catch_bad_pairs(results, threshold), output_src_tar_path)


if __name__ == "__main__":
    main()