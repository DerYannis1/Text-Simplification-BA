import json
import re
import sys
from statistics import mean

WORD_REGEX = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
SENTENCE_REGEX = re.compile(r"[.!?]+")
AOA_FILE = "./lexical_ressources/GPT_estimates_AoA.xlsx"


        self.aoa_lookup = self._load_lexical_resource(AOA_FILE, "Word", "aoa_ft", agg="min")


def words(text):
    return WORD_REGEX.findall(text or "")


def sentences(text):
    return [s for s in SENTENCE_REGEX.split(text or "") if s.strip()]

def load_lexical_resource(cls, path, word_col, value_col, agg="max"):
        
        suffix = Path(path).suffix.lower()
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="cp1252")

        df = df[[word_col, value_col]].dropna()
        df = cls._split_multiword_entries(df, word_col)
        df[word_col] = df[word_col].astype(str).str.strip()
        df = df[df[word_col].str.match(cls._ALPHA_ONLY_RE)]
        df["_key"] = df[word_col].str.lower()

        grouped = df.groupby("_key")[value_col].max() if agg == "max" else df.groupby("_key")[value_col].min()
        return grouped.astype(float).to_dict()


def main():
    infile = "./dataset_mt5/train.jsonl"
    outfile = "./dataset_mt5/stats.txt"

    input_words, klexikon_words, miniklexikon_words = [], [], []
    input_sent_lens, klexikon_sent_lens, miniklexikon_sent_lens = [], [], []
    word_length_source, word_length_miniklexikon, word_length_klexikon = [], [], []
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

                for s in sentences(article.get("input_text", "")):
                    input_sent_lens.append(len(words(s)))
                for s in sentences(article.get("target_text", "")):
                    miniklexikon_sent_lens.append(len(words(s)))
            else:
                n_klexikon += 1

                klexikon_text = words(article.get("target_text", ""))
                klexikon_words.append(len(klexikon_text))
                word_length_klexikon.extend(len(w) for w in klexikon_text)

                for s in sentences(article.get("target_text", "")):
                    klexikon_sent_lens.append(len(words(s)))


        word_compression_miniklexikon = [t / i for i, t in zip(input_words, miniklexikon_words) if i > 0]
        word_compression_klexikon = [t / i for i, t in zip(input_words, klexikon_words) if i > 0]


    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"Anzahl Miniklexikon Datensaetze: {n_miniklexikon}\n")
        f.write(f"Anzahl Klexikon Datensaetze: {n_klexikon}\n")

        f.write(f"Avg. Woerter pro Wikipedia: {mean(input_words):.2f}\n")
        f.write(f"Avg. Woerter pro Miniklexikon: {mean(miniklexikon_words):.2f}\n")
        f.write(f"Avg. Woerter pro Klexikon: {mean(klexikon_words):.2f}\n")

        f.write(f"Avg. Satzlaenge Wikipedia (Woerter/Satz): {mean(input_sent_lens):.2f}\n")
        f.write(f"Avg. Satzlaenge Miniklexikon (Woerter/Satz): {mean(miniklexikon_sent_lens):.2f}\n")
        f.write(f"Avg. Satzlaenge Klexikon (Woerter/Satz): {mean(klexikon_sent_lens):.2f}\n")

        f.write(f"Avg. Wortlaenge Wikipedia (Zeichen/Wort): {mean(word_length_source):.2f}\n")
        f.write(f"Avg. Wortlaenge Miniklexikon (Zeichen/Wort): {mean(word_length_miniklexikon):.2f}\n")
        f.write(f"Avg. Wortlaenge Miniklexikon (Zeichen/Wort): {mean(word_length_klexikon):.2f}\n")

        #AoA

        #WSTF, LIX, FLESH

        #BertScore

        f.write(f"Avg. Kompressionsrate (Miniklexikon/Input): {mean(word_compression_miniklexikon):.3f}\n")
        f.write(f"Avg. Kompressionsrate (Klexikon/Input): {mean(word_compression_klexikon):.3f}\n")


    print(f"Fertig -> {outfile}")


if __name__ == "__main__":
    main()