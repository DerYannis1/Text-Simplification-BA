import argparse
import json
import re
import random
import statistics
from pathlib import Path
from collections import defaultdict

INPUT = Path("dataset_mt5/all.jsonl")
OUTDIR = Path("dataset_mt5")
SEED = 42
WORD_REGEX = re.compile(r"[A-Za-zÄÖÜäöüß]{4,}")

LEVEL_TOKENS = {
    "Klexikon": "<KLEXIKON>",
    "MiniKlexikon": "<MINIKLEXIKON>",
}

FRAGFINN_REGEX = re.compile(
    r"\n?\s*Mehr über\s*[„\"]?[^\n„\"]*[“\"]?\s*findest du auch im "
    r"Kinder-Lexikon Klexikon und bei der Such-Maschine Frag Finn\s*\.?\s*$",
    re.IGNORECASE,
)


def remove_datei_blocks(text):
    """
    Removes wiki markup blocks [[Datei:...]]
    """
    output = []
    position = 0

    while position < len(text):
        is_datei_start = (
            text[position:position + 2] == "[["
            and text[position + 2:position + 8].lower() == "datei:"
        )

        if not is_datei_start:
            output.append(text[position])
            position += 1
            continue

        open_brackets = 1
        position += 2  # skip "[["

        while position < len(text) and open_brackets > 0:
            if text[position:position + 2] == "[[":
                open_brackets += 1
                position += 2
            elif text[position:position + 2] == "]]":
                open_brackets -= 1
                position += 2
            else:
                position += 1

    return "".join(output)


def clean_source(text):
    text = remove_datei_blocks(text)
    text = text.replace("[[", "[")
    text = re.sub(r"\]\s*\]", "]", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_fragfinn(text):
    new_text, count = FRAGFINN_REGEX.subn("", text)
    return new_text.strip(), count > 0


def grounding_score(source, target):
    wiki_words = set(word.lower() for word in WORD_REGEX.findall(source))
    target_words = set(word.lower() for word in WORD_REGEX.findall(target))
    if not target_words:
        return 0.0
    return len(target_words & wiki_words) / len(target_words)


def format_input(source, level):
    return f"vereinfache {LEVEL_TOKENS[level]}: {source}"


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():

    OUTDIR.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(INPUT)

    processed = []
    for r in records:
        wiki_clean = clean_source(r["source"])
        target_clean, _ = remove_fragfinn(r["target"])
        score = grounding_score(wiki_clean, target_clean)

        processed.append({
            "id": r["id"],
            "level": r["level"],
            "input_text": format_input(wiki_clean, r["level"]),
            "target_text": target_clean,
            "grounding_score": round(score, 4),
        })

    # Split auf ID-Ebene (Klexikon + MiniKlexikon bleiben zusammen)
    ids = sorted(set(p["id"] for p in processed))
    random.Random(SEED).shuffle(ids)


    #90% Train, 5% Val, 5% Test
    n = len(ids)
    n_train = int(n * 0.9)
    n_val = int(n * 0.05)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    splits = defaultdict(list)
    for article in processed:
        if article["id"] in train_ids:
            splits["train"].append(article)
        elif article["id"] in val_ids:
            splits["val"].append(article)
        else:
            splits["test"].append(article)

    for name, items in splits.items():
        save_jsonl(OUTDIR / f"{name}.jsonl", items)


if __name__ == "__main__":
    main()