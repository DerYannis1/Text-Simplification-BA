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
MODEL_DIR = "./mt5_hyperparameter_tuning/run03_lr0.0003_nte30_lsf0.1"

AOA_FILE = "./lexical_ressources/GPT_estimates_AoA.xlsx" # Korrelation von 0.96 zu den menschlich erhobenen Werten
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
        "aoa_threshold": 7.5,
        "wstf_threshold": 5.0,
        "lix_threshold": 35.0,
        "min_length_ratio": 0.25
    },
    "2": {
        "control_level": "KLEXIKON",
        "aoa_threshold": 10.0,
        "wstf_threshold": 6.0,
        "lix_threshold": 39.0,
        "min_length_ratio": 0.3
    },
    "3": {
        "control_level": "KLEXIKON",
        "aoa_threshold": 12.0,
        "wstf_threshold": 7.5,
        "lix_threshold": 45.0,
        "min_length_ratio": 0.35
    },
}

GENERATION_MAX_NEW_TOKENS = 512
GENERATION_MIN_LENGTH_FLOOR = 8
GENERATION_MIN_LENGTH_CEILING = 256
GENERATION_NUM_BEAMS = 6
GENERATION_NO_REPEAT_NGRAM_SIZE = 3
GENERATION_REPETITION_PENALTY = 1.15
MAX_SOURCE_LENGTH = 1024
MAX_HALLUCINATION_FIX_ITERATIONS = 3

NO_ERROR = (
    "kein fehler", "ist korrekt", "korrekte aussage", "nicht falsch",
    "keine fehler", "an sich nicht falsch",
)

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
        self.aoa_lookup = self._load_lexical_resource(AOA_FILE, "Word", "aoa_ft", agg="min")
        #self.freq_lookup = self._load_lexical_resource(FREQ_FILE, "Lemma", "Relative Frequenz", agg="max") #hat nut probleme gemacht
        self.language_analyzer = spacy.load(SPACY_MODEL)

    _ALPHA_ONLY_RE = re.compile(r"^[A-Za-zÄÖÜäöüß\-]+$")

    @staticmethod
    def _split_multiword_entries(df, word_col):
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

    @staticmethod
    def _compute_min_length(input_token_count, min_length_ratio):
        target = int(round(input_token_count * min_length_ratio))
        return max(GENERATION_MIN_LENGTH_FLOOR, min(target, GENERATION_MIN_LENGTH_CEILING))

    def simplify(self, text, control_level, min_length_ratio, max_new_tokens=GENERATION_MAX_NEW_TOKENS):
        input_text = f"vereinfache <{control_level.upper()}>: {text}"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", max_length=MAX_SOURCE_LENGTH, truncation=True
        ).to(self.device)
        min_length = self._compute_min_length(inputs["input_ids"].shape[1], min_length_ratio)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                num_beams=GENERATION_NUM_BEAMS,
                no_repeat_ngram_size=GENERATION_NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=GENERATION_REPETITION_PENALTY,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    _CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"} #Wortarten die als schwer geflagged werden können

    def find_difficult_words(self, text, aoa_threshold):
        doc = self.language_analyzer(text)
        flagged, seen = [], set()
        for token in doc:
            if not token.is_alpha or token.pos_ not in self._CONTENT_POS:
                continue
            key = token.text.lower()
            if key in seen:
                continue
            seen.add(key)
            aoa = self.aoa_lookup.get(key)
            reasons = []
            if aoa is not None and aoa > aoa_threshold:
                reasons.append(f"AoA={aoa}")
            if reasons:
                flagged.append({"word": token.text, "reasons": reasons})
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
        words_only = [w["word"] for w in flagged_words]
        prompt = f"""
        Du bekommst einen Text und eine Liste schwieriger Wörter daraus.

        **Deine einzige Aufgabe:** Finde für jedes Wort ein einfacheres, allgemein bekanntes Synonym,
        das im Kontext des Satzes passt. Wenn kein sinnvolles einfacheres Synonym existiert oder das schwierige Wort ein Eigenname ist, 
        gib als Ersatz exakt das Originalwort zurück (keine Umschreibung, keine Erklärung).

        Gib AUSSCHLIESSLICH ein JSON-Objekt zurück, keine Einleitung, kein Markdown:
        {{"replacements": [{{"word": "<Originalwort>", "replacement": "<einfacheres Wort oder Originalwort>"}}]}}

        Wörterliste: {words_only}
        Text (nur als Kontext, NICHT zurückgeben): {input_text}
        """
        raw = self._call_ollama(prompt, temperature=0.0, force_json=True)

        try:
            parsed = json.loads(raw)
            replacements = parsed.get("replacements", [])
        except (json.JSONDecodeError, AttributeError):
            replacements = []

        result = input_text
        for r in replacements:
            word = r.get("word", "")
            replacement = r.get("replacement", "")
            if not word or not replacement or word == replacement:
                continue
            pattern = re.compile(r"(?<!-)\b" + re.escape(word) + r"\b(?!-)")
            result = pattern.sub(lambda m: self._match_case(replacement, m.group()), result)

        return result

    @staticmethod
    def _match_case(replacement, original_match):
        if original_match.isupper():
            return replacement.upper()
        if original_match[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    def detect_hallucinations(self, original_source, input_text):
        prompt = f"""
        Du bekommst:
        - ein Referenzdokument
        - einen einfach geschriebenen Text, der auf dem Referenzdokument basieren soll.

        **Deine einzige Aufgabe:** Finde Sätze im einfachen Text, die einen der folgenden Fehler enthalten:

        1. Widersprüche zum Referenzdokument oder erkennbar erfundene/falsche Fakten
        (z. B. falsche Jahreszahlen, falsche Orte, falsche Namen, falsche Wortherkunft,
        falsche Eigenschaften von Personen/Tieren/Dingen, falsch gesetzte oder fehlende
        Verneinungen, vertauschte/umgekehrte Aussagen).

        2. Unsinnige oder unlogische Aussagen: Sätze, die in sich widersprüchlich sind, keinen
        erkennbaren Sinn ergeben oder inhaltsleere Füllsätze ohne Bezug zum
        eigentlichen Thema sind.

        3. Sachlich falsche Vergleiche oder Analogien: Sätze, die eine Größe, ein Gewicht,
        eine Menge oder Ähnliches mit etwas Alltäglichem vergleichen (z. B. "so schwer wie ein kleines Auto",
        "so groß wie ein Fußballfeld"), bei denen der Vergleichsgegenstand sachlich nicht zutrifft.
        WICHTIG: Solche Vergleiche sind als Stilmittel erwünscht und sollen NICHT als grundsätzlicher
        Fehler gelten -- melde hier ausschließlich Sätze, bei denen der gewählte Vergleichsgegenstand
        falsch oder unpassend ist. Gib in der Begründung (reason) explizit an, dass nur der
        Vergleichsgegenstand ausgetauscht werden soll, nicht der ganze Satz.

        Aussagen, die nicht wörtlich im Referenzdokument stehen, aber auch nicht widersprüchlich,
        unlogisch oder erkennbar falsch sind (allgemein bekanntes, sachlich korrektes Wissen),
        zählen nicht als Fehler. Melde nur Sätze mit einem der drei oben genannten Fehler.

        Gib ausschliesslich ein JSON-Objekt zurück, keine Einleitung, keine Erklärung, kein Markdown:
        {{"has_errors": true/false, "errors": [{{"sentence": "<exakter fehlerhafter Satz>", "reason": "<kurze Begründung>"}}]}}

        Referenzdokument: {original_source}
        Einfacher Text: {input_text}
        """
        raw = self._call_ollama(prompt, temperature=0.0, force_json=True)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or "errors" not in parsed:
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            parsed = {"has_errors": False, "errors": []}
        return parsed

    def correct_hallucinations(self, original_source, input_text, errors):
        error_list_str = "\n".join(
            f'- Satz: "{e}" -> Problem: (nicht spezifiziert)' if isinstance(e, str)
            else f'- Satz: "{e.get("sentence", "")}" -> Problem: {e.get("reason", "")}'
            for e in errors
        )
        prompt = f"""
        Du bekommst ein Referenzdokument und eine Liste fehlerhafter Sätze aus einem einfachen Text,
        mit Begründung.

        **Deine einzige Aufgabe:** Korrigiere jeden gelisteten Satz anhand des Referenzdokuments, so knapp
        wie möglich, im gleichen kurzen, einfachen Sprachstil. Keine zusätzlichen Details, keine
        Zusammenfassungen, keine Umstellungen.

        **Besonderheit bei Vergleichen/Analogien** (z. B. "so schwer wie ein Auto", "so groß wie ein
        Fußballfeld"): Wenn das Problem ein falscher Vergleichsgegenstand ist, LÖSCHE NIEMALS den
        ganzen Vergleich. Ersetze stattdessen nur den Vergleichsgegenstand durch einen sachlich
        passenden, alltagsnahen Gegenstand mit ungefähr der richtigen Größe/dem richtigen Gewicht,
        und behalte die restliche Satzstruktur bei. Beispiel: "Das wiegt so viel wie ein Auto" bei
        einem tatsächlichen Gewicht von 3 kg -> "Das wiegt so viel wie ein Backstein" (nicht ersatzlos
        streichen).

        Gib AUSSCHLIESSLICH ein JSON-Objekt zurück, keine Einleitung, kein Markdown:
        {{"corrections": [{{"original": "<exakter fehlerhafter Satz aus dem Text>", "corrected": "<korrigierter Satz>"}}]}}

        Fehlerhafte Sätze:
        {error_list_str}

        Referenzdokument: {original_source}
        """
        raw = self._call_ollama(prompt, temperature=0.0, force_json=True)

        try:
            parsed = json.loads(raw)
            corrections = parsed.get("corrections", [])
        except (json.JSONDecodeError, AttributeError):
            corrections = []

        result = input_text
        for c in corrections:
            original = c.get("original", "")
            corrected = c.get("corrected", "")
            if not original or not corrected or len(corrected.split()) < 3:
                continue
            if original in result:
                result = result.replace(original, corrected, 1)
                continue
            sentences = re.split(r"(?<=[.!?])\s+", result)
            best = max(sentences, key=lambda s: SequenceMatcher(None, s, original).ratio(), default=None)
            if best and SequenceMatcher(None, best, original).ratio() > 0.75:
                result = result.replace(best, corrected, 1)
        return result


    def _is_real_error(self, error):
        if isinstance(error, str):
            reason = error.lower()
        else:
            reason = (error.get("reason") or "").lower()
        if not reason:
            return False  # ohne Begründung nicht korrigieren
        return not any(marker in reason for marker in NO_ERROR)

    def fix_hallucination(self, original_source, input_text, max_iterations=MAX_HALLUCINATION_FIX_ITERATIONS):
        current_text = input_text
        all_detected_errors = []
        for _ in range(max_iterations):
            detection = self.detect_hallucinations(original_source, current_text)
            errors = [e for e in detection.get("errors", []) if self._is_real_error(e)]
            if not errors:
                break
            all_detected_errors.extend(errors)
            current_text = self.correct_hallucinations(original_source, current_text, errors)
        return current_text, all_detected_errors

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
    
    def _cleanup_text(self, text):
        text = re.sub(r"\s*\.\s*\.", ".", text)         # verwaiste Punkte
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)    # Leerzeichen vor Punkt, etc.
        text = re.sub(r"[ \t]{2,}", " ", text).strip()  #Mehrere Leerzeichen zu einem zusammenfassen
        sentences = re.split(r"(?<=[.!?])\s+", text)    #Satzerlennung
        kept = []
        for s in sentences:
            if any(SequenceMatcher(None, s.lower(), k.lower()).ratio() > 0.85 for k in kept):
                continue
            kept.append(s)
        return " ".join(kept)

    def run(self, source_text, age_group="1", max_content_fix_iterations=2):
        if age_group not in AGE_GROUPS:
            raise ValueError(f"Bitte wähle 1,2 oder 3 als Altersgruppe (Das entspricht ungefähr 1-->5-8, 2-->9-11, 3-->11-14)")
        age_specs = AGE_GROUPS[age_group]
        steps = []

        simplified = self.simplify(source_text, control_level=age_specs["control_level"], min_length_ratio=age_specs["min_length_ratio"])
        steps.append({"step": "1", "name": "mt5_simplification", "input": source_text, "output": simplified})

        content_fixed, detected_hallucinations = self.fix_hallucination(source_text, simplified)

        for i in range(2):
            if(self.may_be_reference(content_fixed, source_text, simplified)):
                simplified = self.simplify(source_text, control_level=age_specs["control_level"], min_length_ratio=age_specs["min_length_ratio"])
                content_fixed, detected_hallucinations = self.fix_hallucination(source_text, simplified)

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
        
        final_text = self._cleanup_text(final_text)

        final_flagged_words = self.find_difficult_words(final_text, age_specs["aoa_threshold"])
        final_readability = self.compute_readability(final_text, age_specs["wstf_threshold"], age_specs["lix_threshold"])
        steps.append({"step": "7", "name": "analyze_final_text", "input": final_text})

        return {
            #delete comments for debugging
            #"source": source_text, 
            #"age_group": age_group, 
            "final_text": final_text,
            #"mt5_simplification": simplified,
            #"content_fixed": content_fixed,
            #"words_fixed": words_fixed,
            #"flagged_words": flagged_words, "readability": readability,
            #"detected_hallucinations": detected_hallucinations,
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