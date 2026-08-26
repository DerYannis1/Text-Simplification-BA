import re
from pathlib import Path
from statistics import mean
import textstat

import pandas as pd
import spacy
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

WORD_REGEX = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
SENTENCE_REGEX = re.compile(r"[.!?]+")
ALPHA_ONLY_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ]+$")
AOA_FILE = "./lexical_ressources/GPT_estimates_AoA.xlsx"

AOA_RELEVANT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
nlp = spacy.load("de_core_news_md", disable=["ner", "parser"])

textstat.set_lang("de")
rouge_l_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

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


def bertscore(candidates, references, lang="de"):
    _, _, f1 = bert_score_fn(candidates, references, lang=lang, verbose=False)
    return f1.mean().item()


def load_texts(path):
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


def collect_stats(bucket, text, aoa_lookup):
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


def write_bucket_stats(f, bucket):
    f.write(f"Avg. Woerter pro Text: {mean(bucket['words']):.2f}\n")
    f.write(f"Avg. Satzlaenge (Woerter/Satz): {mean(bucket['sent_lens']):.2f}\n")
    f.write(f"Avg. Wortlaenge (Zeichen/Wort): {mean(bucket['word_lengths']):.2f}\n")
    f.write(f"Avg. AoA: {mean(bucket['aoa']):.2f}\n")
    f.write(f"Avg. WSTF: {mean(bucket['wstf']):.2f}\n")
    f.write(f"Avg. LIX: {mean(bucket['lix']):.2f}\n")
    f.write(f"Avg. Flesch-Amstad: {mean(bucket['flesch']):.2f}\n")


def main():
    infile = "./human_evaluation/results_G4All.txt"
    outfile = "./human_evaluation/stats_G4All_v2.txt"

    aoa_lookup = load_lexical_resource(AOA_FILE, "Word", "aoa_ft", agg="min")

    texts = load_texts(infile)
    groups = [texts[i:i + 4] for i in range(0, len(texts), 4)]
    groups = [g for g in groups if len(g) == 4]

    source_bucket = new_stat_buckets()
    source_texts_raw = []

    target_buckets = {level: new_stat_buckets() for level in AGE_GROUPS}
    target_texts_raw = {level: [] for level in AGE_GROUPS}
    compression_rates = {level: [] for level in AGE_GROUPS}
    rouge_l_scores = {level: [] for level in AGE_GROUPS}

    for group in groups:
        source_text = group[0]
        collect_stats(source_bucket, source_text, aoa_lookup)
        source_texts_raw.append(source_text)
        source_word_count = len(words(source_text))

        for level, target_text in zip(AGE_GROUPS, group[1:4]):
            collect_stats(target_buckets[level], target_text, aoa_lookup)
            target_texts_raw[level].append(target_text)

            if source_word_count > 0:
                compression_rates[level].append(len(words(target_text)) / source_word_count)

            rouge_l = rouge_l_scorer.score(source_text, target_text)["rougeL"].fmeasure
            rouge_l_scores[level].append(rouge_l)

    bertscores = {level: bertscore(target_texts_raw[level], source_texts_raw) for level in AGE_GROUPS}

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"Anzahl Artikel: {len(groups)}\n\n")

        f.write("--- Wikipedia (Source) ---\n")
        write_bucket_stats(f, source_bucket)
        f.write("\n")

        for level in AGE_GROUPS:
            f.write(f"--- Altersgruppe {level} Jahre (Target) ---\n")
            write_bucket_stats(f, target_buckets[level])
            f.write(f"BERTScore F1 ({level} vs. Wikipedia): {bertscores[level]:.4f}\n")
            f.write(f"Avg. Kompressionsrate ({level}/Wikipedia): {mean(compression_rates[level]):.3f}\n\n")
            f.write(f"Avg. ROUGE-L ({level} vs. Wikipedia): {mean(rouge_l_scores[level]):.4f}\n\n")

    print(f"Fertig -> {outfile}")


if __name__ == "__main__":
    main()