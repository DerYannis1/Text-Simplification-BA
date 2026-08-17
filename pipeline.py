import re, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import textstat
import spacy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from difflib import SequenceMatcher

#MODEL_DIR = "./mt5-simplify-de"
MODEL_DIR = "./mt5-sweep/run03_lr0.0003_nte30_lsf0.1"

AOA_FILE = "./lexical_ressources/GPT_estimates_AoA.xlsx" # Korrelation von 0.96 zu den alten Werten
#AOA_FILE = "./lexical_ressources/Age_Of_Acquisition.csv"
FREQ_FILE = "./lexical_ressources/grundwortschatz.csv"
SPACY_MODEL = "de_core_news_md"
OLLAMA_URL = "http://ollama:11434/api/generate"
OLLAMA_MODEL = "llama4:latest"
OLLAMA_KEEP_ALIVE = "5m"

FREQ_THRESHOLD = 5.0

#1-->5-8
#2-->9-11
#3-->11-14
AGE_GROUPS = {
    "1": {
        "control_level": "MINIKLEXIKON",
        "aoa_threshold": 7.0,
        "wstf_threshold": 5.0,
        "lix_threshold": 35.0,
    },
    "2": {
        "control_level": "KLEXIKON",
        "aoa_threshold": 9.0,
        "wstf_threshold": 6.0,
        "lix_threshold": 39.0,
    },
    "3": {
        "control_level": "KLEXIKON",
        "aoa_threshold": 10.5,
        "wstf_threshold": 7.5,
        "lix_threshold": 45.0,
    },
}

GENERATION_MAX_NEW_TOKENS = 512
GENERATION_MIN_LENGTH = 32
GENERATION_NUM_BEAMS = 6
GENERATION_NO_REPEAT_NGRAM_SIZE = 3
GENERATION_REPETITION_PENALTY = 1.15
MAX_SOURCE_LENGTH = 1024  # muss zu --max-source-length aus train_mt5.py passen

textstat.set_lang("de")


class SimplificationPipeline:
    def __init__(self):
        #subprocess.run(["ollama", "stop", OLLAMA_MODEL], check=False, timeout=30)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        self.model.generation_config.max_length = None
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # agg="min" fuer AoA (frueher erworbene Lesart gewinnt bei Gross-/
        # Kleinschreibungs-Konflikten), agg="max" fuer Frequenz (haeufigere
        # Lesart gewinnt) -- siehe Docstring von _load_lexical_resource.
        self.aoa_lookup = self._load_lexical_resource(AOA_FILE, "Word", "GPT_AoA_ft", agg="min")
        self.freq_lookup = self._load_lexical_resource(FREQ_FILE, "Lemma", "Relative Frequenz", agg="max")
        self.language_analyzer = spacy.load(SPACY_MODEL)

    _ALPHA_ONLY_RE = re.compile(r"^[A-Za-zÄÖÜäöüß\-]+$")

    @staticmethod
    def _split_multiword_entries(df, word_col):
        """Manche Grundwortschatz-Zeilen fassen mehrere Lemmata mit einer
        gemeinsamen Frequenz zusammen (z.B. 'der, die, das' als ein Eintrag
        mit Rang 7). Dadurch waeren die Einzelwoerter 'der'/'die'/'das' -- 
        einige der haeufigsten Woerter ueberhaupt -- im Lookup faktisch nie
        auffindbar, weil find_difficult_words() immer Einzelwoerter extrahiert.
        Solche Zeilen werden hier aufgespalten; jedes Teilwort erbt denselben Wert."""
        if df[word_col].astype(str).str.contains(",").any():
            df = df.copy()
            df[word_col] = df[word_col].astype(str).str.split(",")
            df = df.explode(word_col)
            df[word_col] = df[word_col].str.strip()
        return df

    @classmethod
    def _load_lexical_resource(cls, path, word_col, value_col, agg="max"):
        
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
            key = w.lower()
            if key in seen:
                continue
            seen.add(key)
            aoa = self.aoa_lookup.get(key)
            freq = self.freq_lookup.get(key)
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

    def may_be_reference(self, output, reference, input_text, unwanted_similarity=0.75):
        reference_sim = SequenceMatcher(None, output.lower(), reference.lower()).ratio()
        len_output_input = len(output.split()) / max(len(input_text.split()), 1)
        return reference_sim > unwanted_similarity or len_output_input > 1.5

    def fix_words(self, input_text, flagged_words):
        prompt = f"""
        Du bekommst:
        - einen einfach geschriebenen Text (kurze Sätze, einfache Wörter) 
        - eine Liste von Wörtern, die durch einfachere Synonyme ersetzt werden sollen.

        **Deine einzige Aufgabe:** Ersetze genau die Wörter aus der Liste durch ein einfacheres, allgemein bekanntes und passendes Synonym. Wort für Wort, an derselben Stelle im Satz.

        **Was du NICHT tun darfst:**
        - Kein anderes Wort verändern, auch wenn es dir kompliziert erscheint. Nur die Wörter aus der Liste.
        - Keine Sätze umformulieren, umstellen, kürzen oder verlängern.
        - Keine Wörter in Anführungszeichen ersetzen.
        
        **Bedingungen:**
        - Satzzeichen, Satzstruktur und alle nicht gelisteten Wörter bleiben exakt wie im Original.
        - Wenn für ein Wort kein einfacheres Synonym existiert, lass es unverändert -- ersetze es NICHT durch eine Umschreibung oder Erklärung.
        - Die Wortanzahl der Ausgabe darf sich gegenüber dem Eingabetext um maximal 10% unterscheiden.

        **Beispiel:**
        Wörterliste: [{{"word": "äußerst"}}]
        Eingabetext: "Das ist eine äußerst schwere Aufgabe."
        Korrigierte Ausgabe: "Das ist eine sehr schwere Aufgabe."
        (Nur "äußerst" wurde ersetzt. Alles andere blieb exakt gleich.)

        Wenn keines der gelisteten Wörter ein sinnvolles einfacheres Synonym hat, gib den Eingabetext exakt und unverändert zurück.
        Wenn die Liste leer ist. Gib den Eingabetext exakt zurück. 

        Wörterliste: {flagged_words}
        Eingabetext: {input_text}

        **Gib ausschließlich den überarbeiteten Text zurück. Keine Einleitung, keine Erklärung, keine Anführungszeichen, keine Kommentare von dir. **
        """
        return self._call_ollama(prompt, temperature=0.0)

    def fix_hallucination(self, original_source, input_text):
        prompt = f"""
        Du bekommst: 
        -	einen einfach geschriebenen Text (kurze Sätze, einfache Wörter) 
        -	ein Referenzdokument.

        **Deine einzige Aufgabe:** Prüfe den einfachen Text. Finde nur Sätze, die dem Referenzdokument widersprechen oder erkennbar erfundene/falsche Fakten enthalten (z. B. falsche Jahreszahlen, falsche Orte, falsche Namen, falsche Wortherkunft, falsche Eigenschaften von Personen/Tieren/Dingen). Korrigiere nur diese Sätze, so knapp wie möglich.

        **Was du NICHT tun darfst:**
        - Sätze, die bereits korrekt sind umformulieren oder stilistisch ändern. Übernimm sie Wort für Wort unverändert.
        - Keine Sätze verlängern, keine Nebensätze, Erklärungen oder zusätzlichen Details einfügen.
        - Keine einfachen Wörter durch komplexere/fachlichere ersetzen.
        - Keine Sätze zusammenfassen, aufteilen oder umstellen.
        **Bedinungen:**
        - Aussagen, die nicht wörtlich im Referenzdokument stehen, aber auch nicht widersprüchlich oder erkennbar falsch sind (z. B. allgemein bekanntes Wissen), lässt du unverändert stehen -- das ist KEINE Halluzination, nur fehlende Quellenbelegung, und daher kein Grund zur Änderung.
        - Die Wortanzahl der Ausgabe darf sich gegenüber dem Eingabetext um maximal 10% unterscheiden.
        - NIEMALS das Referenzdokument (auch nicht in Teilen oder nahezu wortwörtlich) als Ausgabe verwenden.
        - Die Ausgabe muss immer im kurzen, einfachen Sprachstil des Eingabetexts bleiben. 
        - Falls der Eingabetext so inhaltlich falsch ist, dass eine satzweise Korrektur nicht sinnvoll möglich ist: korrigiere so viele einzelne falsche Aussagen wie du sinnvoll kannst, aber erschaffe keinen neuen, komplexeren Text aus dem Referenzdokument.

        **Beispiel:**
        Referenzdokument: „Albert Einstein wurde 1879 in Ulm geboren. Später lebte er unter anderem in der Schweiz und in den USA.“
        Eingabetext: „Albert Einstein wurde in Ulm geboren. Er kam 1889 zur Welt. Später lebte er in der Schweiz und in den USA.“
        Korrigierte Ausgabe: „Albert Einstein wurde in Ulm geboren. Er kam 1879 zur Welt. Später lebte er in der Schweiz und in den USA.“
        (Nur die falsche Jahreszahl "1889" wurde durch „1879“ ersetzt. Die anderen Sätze blieben exakt gleich. Keine Umformulierung, keine Ergänzung.)

        Wenn der komplette Eingabetext schon korrekt ist, gib ihn exakt unverändert zurück.

        Referenzdokument: {original_source}
        Eingabetext: {input_text}

        **Gib ausschließlich den (gegeben falls) überarbeiteten Text zurück. Keine Einleitung, keine Erklärung, keine Anführungszeichen, keine Kommentare von dir. **
        """
        return self._call_ollama(prompt, temperature=0.0)

    def fix_readability(self, input_text):
        prompt = f"""Du bekommst: 
        -	einen einfach geschriebenen Text, dessen Sätze für die Zielgruppe noch zu lang oder zu verschachtelt sind.

        **Deine einzige Aufgabe:** Teile lange oder verschachtelte Sätze in mehrere kurze, einfache Sätze auf, wo immer das sinnvoll möglich ist.

        **Was du NICHT tun darfst:**
        - Kein Wort durch ein anderes ersetzen. Die Wortwahl bleibt exakt wie im Original. Ausnahme: ein Bezugswort (z. B. ein wiederholtes Subjekt wie "Sie" oder "Das") darf beim Aufteilen eines Satzes wiederholt werden, das ist okay.
        - Keine Informationen hinzufügen oder entfernen.
        - Keine Sätze zusammenfassen, kürzen oder verdichten. Nur aufteilen, niemals das Gegenteil.

        **Bedingungen**
        - Sätze, die schon kurz und einfach sind, bleiben unverändert. Nicht jeder Satz muss geteilt werden.
        - Die Reihenfolge der Informationen bleibt erhalten.
        - Die Wortanzahl der Ausgabe darf sich gegenüber dem Eingabetext um maximal 15% unterscheiden

        **Beispiel:**
        Eingabetext: „Wenn Kinder regelmäßig Sport treiben, sind sie oft fitter und können sich beim Lernen besser konzentrieren.“
        Aufgeteilte Ausgabe: „Manche Kinder treiben regelmäßig Sport. Sie sind oft fitter. Sie können sich beim Lernen besser konzentrieren.“
        (Ein langer Satz mit zwei Nebenbedingungen wurde in drei kurze Sätze aufgeteilt. Kein Wort wurde ersetzt, keine Information ging verloren oder kam hinzu.)

        Wenn der Text schon durchgehend aus kurzen, einfachen Sätzen besteht, gib ihn exakt unverändert zurück.

        Eingabetext: {input_text}

        **Gib ausschließlich den (gegeben falls) überarbeiteten Text zurück. Keine Einleitung, keine Erklärung, keine Anführungszeichen, keine Kommentare von dir. **

        """
        return self._call_ollama(prompt, temperature=0.0)

    @staticmethod
    def _call_ollama(prompt, temperature=0.2, force_json=False):
        payload = {
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature}, "keep_alive": OLLAMA_KEEP_ALIVE,
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

        for i in range(2):
            if(self.may_be_reference(content_fixed, source_text, simplified)):
                simplified = self.simplify(source_text, control_level=age_specs["control_level"])
                content_fixed = self.fix_hallucination(source_text, simplified)

        steps.append({"step": "2", "name": "fix_hallucination", "input": simplified, "raw_llm_response": content_fixed})

        flagged_words = self.find_difficult_words(content_fixed, age_specs["aoa_threshold"])
        steps.append({"step": "3", "name": "find_difficult_words", "output": flagged_words})

        if flagged_words:
            words_fixed = self.fix_words(content_fixed, flagged_words)
            steps.append({"step": "4", "name": "fix_words", "input": content_fixed, "flagged_words_addressed": flagged_words,
                        "output": words_fixed})
        else:
            words_fixed = content_fixed
            steps.append({"step": "4", "name": "SKIPPED: fix_words"})

        readability = self.compute_readability(words_fixed, age_specs["wstf_threshold"], age_specs["lix_threshold"])
        steps.append({"step": "5", "name": "compute_readability", "output": readability})

        if readability["wstf_too_high"] or readability["lix_too_high"]:
            final_text = self.fix_readability(words_fixed)
            steps.append({"step": "6", "name": "fix_readability", "input": words_fixed, "output": final_text})
        else:
            final_text = words_fixed
            steps.append({"step": "6", "name": "SKIPPED: fix_readability"})

        final_flagged_words = self.find_difficult_words(final_text, age_specs["aoa_threshold"])
        final_readability = self.compute_readability(final_text, age_specs["wstf_threshold"], age_specs["lix_threshold"])
        steps.append({"step": "7", "name": "analyze_final_text", "input": final_text})

        return {
            #delete comments for debugging
            #"source": source_text, 
            #"age_group": age_group, 
            "final_text": final_text,
            "mt5_simplification": simplified,
            #"content_fixed": content_fixed,
            #"words_fixed": words_fixed,
            #"flagged_words": flagged_words, "readability": readability,
            #"final_flagged_words": final_flagged_words, "final_readability": final_readability,
            #"steps": steps
        }


if __name__ == "__main__":
    pipeline = SimplificationPipeline()
    example_text = 'Dass ausgerechnet jenes Werkzeug, mit dem auf die Schönheit deutschsprachiger Literatur hingewiesen werden soll, in fragwürdigem Deutsch "Longlist" genannt wird - darüber regt sich längst keiner mehr auf. Entsprechende Mäkeleien aus der Anfangszeit des Deutschen Buchpreises vor über 20 Jahren sind inzwischen verstummt. Der Preis und seine Listen - die "Longlist" im August mit 20 Titeln und die auf sechs Romane verkürzte "Shortlist" im September - sind etabliert und feste Größen im Kalender des Literaturbetriebs.Ärger, Empörung und Polemik lösen sie trotzdem regelmäßig und zuverlässig aus - und erfüllen damit zielgenau den Zweck des Deutschen Buchpreises. Denn Streit schafft Aufmerksamkeit - die wichtigste Währung auf dem hart umkämpften Buchmarkt, auf dem immer mehr Titel um die Gunst eines stetig kleiner werdenden Publikums ringen.'
    for age_group in AGE_GROUPS:
        result = pipeline.run(example_text, age_group=age_group)
        with open(f"pipeline_output_{age_group}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)