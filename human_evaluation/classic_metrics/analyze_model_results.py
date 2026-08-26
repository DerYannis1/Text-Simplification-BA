import re
from pathlib import Path
from statistics import mean
import textstat

import pandas as pd
import spacy

WORD_REGEX = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
SENTENCE_REGEX = re.compile(r"[.!?]+")
ALPHA_ONLY_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ]+$")
AOA_FILE = "./lexical_ressources/GPT_estimates_AoA.xlsx"

# AoA-Normen nur für Inhaltswörter
AOA_RELEVANT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
nlp = spacy.load("de_core_news_md", disable=["ner", "parser"])

textstat.set_lang("de")

# Reihenfolge der Zeilen innerhalb einer 4er-Gruppe: Zeile 0 ist das
# Wikipedia-Original (separat schon analysiert, hier ausgelassen), Zeilen 1-3
# sind die Vereinfachungen fuer die drei Altersgruppen.
AGE_GROUPS = ["5-8", "8-11", "11-14"]


def words(text):
    return WORD_REGEX.findall(text or "")


def sentences(text):
    return [s for s in SENTENCE_REGEX.split(text or "") if s.strip()]


def content_words(text):
    doc = nlp(text or "")
    return [tok.text for tok in doc if tok.is_alpha and tok.pos_ in AOA_RELEVANT_POS]


def load_lexical_resource(path, word_col, value_col, agg="max"):
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp1252")

    df = df[[word_col, value_col]].dropna()
    df[word_col] = df[word_col].astype(str).str.strip()
    df = df[df[word_col].str.match(ALPHA_ONLY_RE)]
    df["_key"] = df[word_col].str.lower()

    grouped = df.groupby("_key")[value_col].max() if agg == "max" else df.groupby("_key")[value_col].min()
    return grouped.astype(float).to_dict()


def aoa_values(word_list, lookup):
    return [lookup[w.lower()] for w in word_list if w.lower() in lookup]


def compute_readability(text):
    wstf = textstat.wiener_sachtextformel(text, variant=1)
    lix = textstat.lix(text)
    flesch_amstad = textstat.flesch_reading_ease(text)
    return {"wstf": wstf, "lix": lix, "flesch_amstad": flesch_amstad}


def load_texts(path):
    # Jede Zeile ist ein in Anfuehrungszeichen gesetzter Text, keine JSONL.
    texts = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            texts.append(line)
    return texts


def new_stat_buckets():
    return {"words": [], "sent_lens": [], "word_lengths": [], "aoa": [], "wstf": [], "lix": [], "flesch": []}


def main():
    infile = "./human_evaluation/results_BA.txt"
    outfile = "./human_evaluation/stats_BA.txt"

    aoa_lookup = load_lexical_resource(AOA_FILE, "Word", "aoa_ft", agg="min")

    texts = load_texts(infile)
    groups = [texts[i:i + 4] for i in range(0, len(texts), 4)]

    stats = {level: new_stat_buckets() for level in AGE_GROUPS}

    for group in groups:
        if len(group) < 4:
            continue  # unvollstaendige letzte Gruppe ueberspringen

        # group[0] ist das Wikipedia-Original -- bereits separat analysiert
        for level, text in zip(AGE_GROUPS, group[1:4]):
            bucket = stats[level]

            text_words = words(text)
            bucket["words"].append(len(text_words))
            bucket["word_lengths"].extend(len(w) for w in text_words)
            bucket["aoa"].extend(aoa_values(content_words(text), aoa_lookup))

            readability = compute_readability(text)
            bucket["wstf"].append(readability["wstf"])
            bucket["lix"].append(readability["lix"])
            bucket["flesch"].append(readability["flesch_amstad"])

            for s in sentences(text):
                bucket["sent_lens"].append(len(words(s)))

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"Anzahl Artikel: {len(groups)}\n\n")

        for level in AGE_GROUPS:
            bucket = stats[level]
            f.write(f"--- Altersgruppe {level} Jahre ---\n")
            f.write(f"Avg. Woerter pro Text: {mean(bucket['words']):.2f}\n")
            f.write(f"Avg. Satzlaenge (Woerter/Satz): {mean(bucket['sent_lens']):.2f}\n")
            f.write(f"Avg. Wortlaenge (Zeichen/Wort): {mean(bucket['word_lengths']):.2f}\n")
            f.write(f"Avg. AoA: {mean(bucket['aoa']):.2f}\n")
            f.write(f"Avg. WSTF: {mean(bucket['wstf']):.2f}\n")
            f.write(f"Avg. LIX: {mean(bucket['lix']):.2f}\n")
            f.write(f"Avg. Flesch-Amstad: {mean(bucket['flesch']):.2f}\n\n")

    print(f"Fertig -> {outfile}")


if __name__ == "__main__":
    main()