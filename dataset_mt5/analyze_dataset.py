import json
import re
import sys
from pathlib import Path
from statistics import mean
import textstat

import pandas as pd
import spacy

WORD_REGEX = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
SENTENCE_REGEX = re.compile(r"[.!?]+")
ALPHA_ONLY_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ]+$")
AOA_FILE = "./lexical_ressources/GPT_estimates_AoA.xlsx"

# AoA-Normen sind nur fuer Inhaltswoerter sinnvoll (Funktionswoerter wie
# Artikel, Pronomen, Praepositionen haben i. d. R. keine verlaesslichen Werte).
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
    flesch-amstad = textstat.flesch_reading_ease(text)
    return {
        "wstf": wstf, "lix": lix, "flesch-amstad": flesch-amstad
    }

def bertscore():
    pass


def main():
    infile = "./dataset_mt5/train.jsonl"
    outfile = "./dataset_mt5/stats.txt"

    aoa_lookup = load_lexical_resource(AOA_FILE, "Word", "aoa_ft", agg="min")

    input_words, klexikon_words, miniklexikon_words = [], [], []
    input_sent_lens, klexikon_sent_lens, miniklexikon_sent_lens = [], [], []
    word_length_source, word_length_miniklexikon, word_length_klexikon = [], [], []
    aoa_source, aoa_miniklexikon, aoa_klexikon = [], [], []
    n_miniklexikon = 0
    n_klexikon = 0

    with open(infile, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)

            if(article.get("level", "") == "MiniKlexikon"):
                n_miniklexikon += 1

                input_text = words(article.get("input_text", ""))
                miniklexikon_text = words(article.get("target_text", ""))
                input_words.append(len(input_text))
                miniklexikon_words.append(len(miniklexikon_text))
                word_length_source.extend(len(w) for w in input_text)
                word_length_miniklexikon.extend(len(w) for w in miniklexikon_text)
                aoa_source.extend(aoa_values(content_words(article.get("input_text", "")), aoa_lookup))
                aoa_miniklexikon.extend(aoa_values(content_words(article.get("target_text", "")), aoa_lookup))

                for s in sentences(article.get("input_text", "")):
                    input_sent_lens.append(len(words(s)))
                for s in sentences(article.get("target_text", "")):
                    miniklexikon_sent_lens.append(len(words(s)))
            else:
                n_klexikon += 1

                klexikon_text = words(article.get("target_text", ""))
                klexikon_words.append(len(klexikon_text))
                word_length_klexikon.extend(len(w) for w in klexikon_text)
                aoa_klexikon.extend(aoa_values(content_words(article.get("target_text", "")), aoa_lookup))

                for s in sentences(article.get("target_text", "")):
                    klexikon_sent_lens.append(len(words(s)))


        word_compression_miniklexikon = [t / i for i, t in zip(input_words, miniklexikon_words) if i > 0]
        word_compression_klexikon = [t / i for i, t in zip(input_words, klexikon_words) if i > 0]


    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"Anzahl Miniklexikon Datensaetze: {n_miniklexikon}\n")
        f.write(f"Anzahl Klexikon Datensaetze: {n_klexikon}\n\n")

        f.write(f"Avg. Woerter pro Wikipedia: {mean(input_words):.2f}\n")
        f.write(f"Avg. Woerter pro Miniklexikon: {mean(miniklexikon_words):.2f}\n")
        f.write(f"Avg. Woerter pro Klexikon: {mean(klexikon_words):.2f}\n\n")

        f.write(f"Avg. Satzlaenge Wikipedia (Woerter/Satz): {mean(input_sent_lens):.2f}\n")
        f.write(f"Avg. Satzlaenge Miniklexikon (Woerter/Satz): {mean(miniklexikon_sent_lens):.2f}\n")
        f.write(f"Avg. Satzlaenge Klexikon (Woerter/Satz): {mean(klexikon_sent_lens):.2f}\n\n")

        f.write(f"Avg. Wortlaenge Wikipedia (Zeichen/Wort): {mean(word_length_source):.2f}\n")
        f.write(f"Avg. Wortlaenge Miniklexikon (Zeichen/Wort): {mean(word_length_miniklexikon):.2f}\n")
        f.write(f"Avg. Wortlaenge Klexikon (Zeichen/Wort): {mean(word_length_klexikon):.2f}\n\n")

        f.write(f"Avg. AoA Wikipedia: {mean(aoa_source):.2f}\n")
        f.write(f"Avg. AoA Miniklexikon: {mean(aoa_miniklexikon):.2f}\n")
        f.write(f"Avg. AoA Klexikon: {mean(aoa_klexikon):.2f}\n\n")


        #WSTF, LIX, FLESH

        #BertScore

        f.write(f"Avg. Kompressionsrate (Miniklexikon/Input): {mean(word_compression_miniklexikon):.3f}\n")
        f.write(f"Avg. Kompressionsrate (Klexikon/Input): {mean(word_compression_klexikon):.3f}\n")


    print(f"Fertig -> {outfile}")


if __name__ == "__main__":
    main()