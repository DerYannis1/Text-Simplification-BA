import re, json, subprocess
import numpy as np
import pandas as pd
import requests
import textstat
import spacy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = "./mt5-simplify-de"

AOA_FILE = "./lexical_ressources/Age_Of_Acquisition.csv"
FREQ_FILE = "./lexical_ressources/grundwortschatz.csv"
SPACY_MODEL = "de_core_news_md"
OLLAMA_URL = "http://ollama:11434/api/generate"
OLLAMA_MODEL = "llama4:latest"

FREQ_THRESHOLD = 5.0

#1-->5-8
#2-->9-11
#3-->11-14
AGE_GROUPS = {
    "1": {
        "control_level": "MINIKLEXIKON",
        "aoa_threshold": 7.0,
        "wstf_threshold": 6.0,
        "lix_threshold": 30.0,
    },
    "2": {
        "control_level": "KLEXIKON",
        "aoa_threshold": 9.0,
        "wstf_threshold": 8.0,
        "lix_threshold": 40.0,
    },
    "3": {
        "control_level": "KLEXIKON",
        "aoa_threshold": 12.0,
        "wstf_threshold": 10.0,
        "lix_threshold": 50.0,
    },
}

GENERATION_MAX_NEW_TOKENS = 512
GENERATION_MIN_LENGTH = 32
GENERATION_NUM_BEAMS = 4
GENERATION_NO_REPEAT_NGRAM_SIZE = 3
GENERATION_REPETITION_PENALTY = 1.3
MAX_SOURCE_LENGTH = 1024

textstat.set_lang("de")


class SimplificationPipeline:
    def __init__(self):
        #subprocess.run(["ollama", "stop", OLLAMA_MODEL], check=False, timeout=30)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.aoa_lookup = self._load_lexical_resource(AOA_FILE, "Word", "AoAestimate")
        self.freq_lookup = self._load_lexical_resource(FREQ_FILE, "Lemma", "Relative Frequenz")
        self.language_analyzer = spacy.load(SPACY_MODEL)

    @staticmethod
    def _load_lexical_resource(path, word_col, value_col):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp1252")
        lookup = {}
        for _, row in df.iterrows():
            if pd.isna(row[value_col]):
                continue
            lookup[str(row[word_col]).strip().lower()] = float(row[value_col])
        return lookup

    def simplify(self, text, control_level, max_new_tokens=GENERATION_MAX_NEW_TOKENS):

        input_text = f"vereinfache <{control_level.upper()}>: {text}"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", max_length=MAX_SOURCE_LENGTH, truncation=True
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_length=GENERATION_MIN_LENGTH,
                num_beams=GENERATION_NUM_BEAMS,
                no_repeat_ngram_size=GENERATION_NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=GENERATION_REPETITION_PENALTY,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def find_difficult_words(self, text, aoa_threshold):
        words = re.findall(r"[A-Za-zÄÖÜäöüß]+", text) #Alle Wörter die mit Buchstaben aus dem Alphabet und ä, ö, ü, ß bestehen
        flagged, seen = [], set()
        for w in words:
            if w in seen:
                continue
            seen.add(w)
            aoa = self.aoa_lookup.get(w)
            freq = self.freq_lookup.get(w)
            reasons = []
            if aoa is not None and aoa > aoa_threshold:
                reasons.append(f"AoA={aoa}")
            if freq is not None and freq < FREQ_THRESHOLD:
                reasons.append(f"Frequenz={freq:.2f}")
            if reasons:
                flagged.append({"word": w, "reasons": reasons})
        return flagged

    def compute_readability(self, text, wstf_threshold, lix_threshold):
        wstf = textstat.wiener_sachtextformel(text, variant=1)
        lix = textstat.lix(text)
        return {
            "wstf": wstf, "lix": lix,
            "wstf_too_high": bool(wstf > wstf_threshold),
            "lix_too_high": bool(lix > lix_threshold),
        }


    def fix_words(self, input_text, flagged_words):
        #schwere Wörter mit geeigneten synonymen ersetzen oder erklären
        prompt = f"""
        Du erhältst: 
        - einen Eingabetext 
        - eine Liste mit Wörtern, die im Eingabetext enthalten sind 
        
        Deine Aufgabe ist es, ausschließlich die Wörter aus der Liste im Eingabetext durch einfachere und allgemein verständlichere Synonyme zu ersetzen. Dabei gelten folgende Regeln: 
        - Verändere nur Wörter, die in der vorgegebenen Liste enthalten sind.
        - Ersetze jedes aufgeführte Wort durch ein einfacheres Synonym, sofern dies im jeweiligen Kontext sinnvoll möglich ist.
        - Behalte die Bedeutung des ursprünglichen Textes vollständig bei.
        - Behalte Grammatik, Satzstruktur, Stil, Formatierung und alle nicht aufgeführten Wörter möglichst unverändert bei.
        - Verändere keine Eigennamen, Fachbegriffe oder andere Wörter, die nicht in der Liste enthalten sind.
        - Wenn für ein Wort kein geeignetes einfacheres Synonym existiert, lasse es unverändert.
        - Füge keine neuen Informationen hinzu und entferne keine Informationen. Gib ausschließlich den überarbeiteten Eingabetext zurück.

        Wörterliste: {flagged_words}
        Eingabetext: {input_text}

        Gib nur den überarbeiteten Text zurück.
        """
        return self._call_ollama(prompt)

    def fix_hallucination(self, original_source, input_text):
        prompt = f"""
        Du erhältst:
        - Ein Referenzdokument
        - Einen Eingabetext, der Fehler wie erfundene Inhalte, Widersprüche, Halluzinationen oder übermäßige Details enthalten kann

        Deine Aufgabe ist es, den Eingabetext so zu überarbeiten, dass er vollständig auf dem Referenzdokument basiert und dabei den Stil des Eingabetexts beibehält. 
        
        Die korrigierte Version muss: 
        - sachlich mit dem Referenzdokument übereinstimmen 
        - keine irrelevanten oder ungenauen Informationen enthalten

        Geben Sie bei Bedarf nur die korrigierte Version des Eingabetextes zurück, andernfalls geben Sie denselben Eingabetext zurück.  
        Referenzdokument: {original_source} 
        Eingabetext: {input_text} 
        Gib nur den überarbeiteten Text zurück.
        """
        return self._call_ollama(prompt, temperature=0.0)

    @staticmethod
    def _call_ollama(prompt, temperature=0.2, force_json=False):
        payload = {
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature}, "keep_alive": 0,
        }
        if force_json:
            payload["format"] = "json"
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["response"].strip()

    def run(self, source_text, age_group="1", max_content_fix_iterations=2):
        if age_group not in AGE_GROUPS:
            raise ValueError(f"Bitte wähle 1,2 oder 3 als Altersgruppe (Das entspricht ungefähr 1-->5-8, 2-->9-11, 3-->11-14)")
        age_specs = AGE_GROUPS[age_group]
        steps = []

        simplified = self.simplify(source_text, control_level=age_specs["control_level"])
        steps.append({"step": "1", "name": "mt5_simplification", "input": source_text, "output": simplified})

        content_fixed = self.fix_hallucination(source_text, simplified)
        steps.append({"step": "2", "name": "fix_hallucination", "input": simplified, "raw_llm_response": content_fixed})

        flagged_words = self.find_difficult_words(content_fixed, age_specs["aoa_threshold"])
        steps.append({"step": "3", "name": "find_difficult_words", "output": flagged_words})

        readability = self.compute_readability(content_fixed, age_specs["wstf_threshold"], age_specs["lix_threshold"])
        steps.append({"step": "4", "name": "compute_readability", "output": readability})

        if flagged_words:
            final_text = self.fix_words(content_fixed, flagged_words)
            steps.append({"step": "5", "name": "fix_words", "input": content_fixed, "flagged_words_addressed": flagged_words,
                        "output": final_text})
        else:
            final_text = content_fixed
            steps.append({"step": "5", "name": "SKIPPED: fix_words"})

        final_flagged_words = self.find_difficult_words(final_text, age_specs["aoa_threshold"])
        final_readability = self.compute_readability(final_text, age_specs["wstf_threshold"], age_specs["lix_threshold"])
        steps.append({"step": "6", "name": "analyze_final_text", "input": final_text})

        return {
            "source": source_text, "age_group": age_group, "final_text": final_text,
            "mt5_simplification": simplified,
            "content_fixed": content_fixed,
            "flagged_words": flagged_words, "readability": readability,
            "final_flagged_words": final_flagged_words, "final_readability": final_readability,
            "steps": steps
        }


if __name__ == "__main__":
    pipeline = SimplificationPipeline()
    example_text = "Paul Attwood studierte von 1988 bis 1991 an der University of Plymouth Bauingenieurwesen und diente später als Soldat bei der Royal Marines. Nachdem Pilot Mark Tout 1996 positiv auf Anabole Steroide getestet wurde, wechselte Attwood in den Bob von Pilot Sean Olsson. Die neuformierte Crew konnte bei der Weltmeisterschaft 1997 den vierten Platz erreichen. Bei den Olympischen Winterspielen 1998 in Nagano gewannen Sean Olsson und Paul Attwood zusammen mit Courtney Rumbolt und Dean Ward die Bronzemedaille im Viererbob-Wettkampf auf der Spiral. Bei seiner zweiten Olympiateilnahme in Salt Lake City 2002 belegte er mit Phil Harries, Dave McCalla und Pilot Lee Johnston den 14. Platz."
    for age_group in AGE_GROUPS:
        result = pipeline.run(example_text, age_group=age_group)
        with open(f"pipeline_output_{age_group}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)