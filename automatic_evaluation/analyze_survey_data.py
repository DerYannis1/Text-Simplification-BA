#Abgewandelte Form von dataset_mt5/analyse_dataset.py

import json
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


def main():
    infile = "./human_evaluation/source_text.jsonl"
    outfile = "./human_evaluation/stats_test_data.txt"

    aoa_lookup = load_lexical_resource(AOA_FILE, "Word", "aoa_ft", agg="min")

    article_words, sent_lens, word_lengths, aoa_scores = [], [], [], []
    wstf_scores, lix_scores, flesch_scores = [], [], []

    n_articles = 0

    with open(infile, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            n_articles += 1

            text = article.get("text", "")
            text_words = words(text)
            article_words.append(len(text_words))
            word_lengths.extend(len(w) for w in text_words)
            aoa_scores.extend(aoa_values(content_words(text), aoa_lookup))

            readability = compute_readability(text)
            wstf_scores.append(readability["wstf"])
            lix_scores.append(readability["lix"])
            flesch_scores.append(readability["flesch_amstad"])

            for s in sentences(text):
                sent_lens.append(len(words(s)))

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"Anzahl Artikel: {n_articles}\n\n")

        f.write(f"Avg. Woerter pro Artikel: {mean(article_words):.2f}\n")
        f.write(f"Avg. Satzlaenge (Woerter/Satz): {mean(sent_lens):.2f}\n")
        f.write(f"Avg. Wortlaenge (Zeichen/Wort): {mean(word_lengths):.2f}\n\n")

        f.write(f"Avg. AoA: {mean(aoa_scores):.2f}\n\n")

        f.write(f"Avg. WSTF: {mean(wstf_scores):.2f}\n")
        f.write(f"Avg. LIX: {mean(lix_scores):.2f}\n")
        f.write(f"Avg. Flesch-Amstad: {mean(flesch_scores):.2f}\n")

    print(f"Fertig -> {outfile}")


if __name__ == "__main__":
    main()