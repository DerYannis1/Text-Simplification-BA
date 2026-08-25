import json
import re
import sys
from pathlib import Path
from statistics import mean
import textstat

import pandas as pd
import spacy
from bert_score import score as bert_score_fn

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
    flesch_amstad = textstat.flesch_reading_ease(text)
    return {"wstf": wstf, "lix": lix, "flesch_amstad": flesch_amstad}


def bertscore(candidates, references, lang="de"):
    _, _, f1 = bert_score_fn(candidates, references, lang=lang, verbose=False)
    return f1.mean().item()


def main():
    infile = "./dataset_mt5/train.jsonl"
    outfile = "./dataset_mt5/stats.txt"

    aoa_lookup = load_lexical_resource(AOA_FILE, "Word", "aoa_ft", agg="min")

    input_words, klexikon_words, miniklexikon_words = [], [], []
    input_sent_lens, klexikon_sent_lens, miniklexikon_sent_lens = [], [], []
    word_length_source, word_length_miniklexikon, word_length_klexikon = [], [], []
    aoa_source, aoa_miniklexikon, aoa_klexikon = [], [], []

    wstf_source, wstf_miniklexikon, wstf_klexikon = [], [], []
    lix_source, lix_miniklexikon, lix_klexikon = [], [], []
    flesch_source, flesch_miniklexikon, flesch_klexikon = [], [], []

    input_texts_raw, miniklexikon_texts_raw, klexikon_texts_raw = [], [], []

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

                input_raw = article.get("input_text", "")
                miniklexikon_raw = article.get("target_text", "")

                input_text = words(input_raw)
                miniklexikon_text = words(miniklexikon_raw)
                input_words.append(len(input_text))
                miniklexikon_words.append(len(miniklexikon_text))
                word_length_source.extend(len(w) for w in input_text)
                word_length_miniklexikon.extend(len(w) for w in miniklexikon_text)
                aoa_source.extend(aoa_values(content_words(input_raw), aoa_lookup))
                aoa_miniklexikon.extend(aoa_values(content_words(miniklexikon_raw), aoa_lookup))

                readability_source = compute_readability(input_raw)
                wstf_source.append(r_source["wstf"])
                lix_source.append(r_source["lix"])
                flesch_source.append(r_source["flesch_amstad"])

                readability_mini = compute_readability(miniklexikon_raw)
                wstf_miniklexikon.append(r_mini["wstf"])
                lix_miniklexikon.append(r_mini["lix"])
                flesch_miniklexikon.append(r_mini["flesch_amstad"])

                input_texts_raw.append(input_raw)
                miniklexikon_texts_raw.append(miniklexikon_raw)

                for s in sentences(input_raw):
                    input_sent_lens.append(len(words(s)))
                for s in sentences(miniklexikon_raw):
                    miniklexikon_sent_lens.append(len(words(s)))
            else:
                n_klexikon += 1

                klexikon_raw = article.get("target_text", "")
                klexikon_text = words(klexikon_raw)
                klexikon_words.append(len(klexikon_text))
                word_length_klexikon.extend(len(w) for w in klexikon_text)
                aoa_klexikon.extend(aoa_values(content_words(klexikon_raw), aoa_lookup))

                readability_klex = compute_readability(klexikon_raw)
                wstf_klexikon.append(r_klex["wstf"])
                lix_klexikon.append(r_klex["lix"])
                flesch_klexikon.append(r_klex["flesch_amstad"])

                klexikon_texts_raw.append(klexikon_raw)

                for s in sentences(klexikon_raw):
                    klexikon_sent_lens.append(len(words(s)))


        word_compression_miniklexikon = [t / i for i, t in zip(input_words, miniklexikon_words) if i > 0]
        word_compression_klexikon = [t / i for i, t in zip(input_words, klexikon_words) if i > 0]

        bertscore_miniklexikon = bertscore(miniklexikon_texts_raw, input_texts_raw)
        bertscore_klexikon = bertscore(klexikon_texts_raw, input_texts_raw)


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

        f.write(f"Avg. WSTF Wikipedia: {mean(wstf_source):.2f}\n")
        f.write(f"Avg. WSTF Miniklexikon: {mean(wstf_miniklexikon):.2f}\n")
        f.write(f"Avg. WSTF Klexikon: {mean(wstf_klexikon):.2f}\n\n")

        f.write(f"Avg. LIX Wikipedia: {mean(lix_source):.2f}\n")
        f.write(f"Avg. LIX Miniklexikon: {mean(lix_miniklexikon):.2f}\n")
        f.write(f"Avg. LIX Klexikon: {mean(lix_klexikon):.2f}\n\n")

        f.write(f"Avg. Flesch-Amstad Wikipedia: {mean(flesch_source):.2f}\n")
        f.write(f"Avg. Flesch-Amstad Miniklexikon: {mean(flesch_miniklexikon):.2f}\n")
        f.write(f"Avg. Flesch-Amstad Klexikon: {mean(flesch_klexikon):.2f}\n\n")

        f.write(f"BERTScore F1 (Miniklexikon vs. Wikipedia): {bertscore_miniklexikon:.4f}\n")
        f.write(f"BERTScore F1 (Klexikon vs. Wikipedia): {bertscore_klexikon:.4f}\n\n")

        f.write(f"Avg. Kompressionsrate (Miniklexikon/Input): {mean(word_compression_miniklexikon):.3f}\n")
        f.write(f"Avg. Kompressionsrate (Klexikon/Input): {mean(word_compression_klexikon):.3f}\n")


    print(f"Fertig -> {outfile}")


if __name__ == "__main__":
    main()