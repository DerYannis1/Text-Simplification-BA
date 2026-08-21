import json
import os
from pathlib import Path
from datetime import datetime

from openai import OpenAI

SOURCE_TEXT_PATH = "./human_evaluation/source_text.jsonl"
OUTPUT_DIR = "./human_evaluation"
MODEL = "gpt-5.6-terra"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

AGE_GROUP_DESCRIPTIONS = {
    "1": """
    Zielgruppe: Kinder von 5 bis 8 Jahren
    Merkmale:
    - Sehr kurze, einfache Sätze
    - Wortschatz der Altersklasse entsprechend
    - Keine Fachbegriffe, keine Fremdwörter, keine verschachtelten Sätze
    - Konkrete, anschauliche Beispiele statt abstrakter Erklärungen
    - Wichtige Informationen sollen möglichst vollständig erhalten bleiben und nur so weit wie für die Zielgruppe nötig gekürzt/umschrieben werden
    """,
    "2": """
    Zielgruppe: Kinder von 8 bis 11 Jahren
    Merkmale:
    - Kurze bis mittellange Sätze
    - Einfacher bis mittlerer Wortschatz
    - Fachbegriffe sind erlaubt, müssen aber im Text kurz erklärt werden
    - Ein roter Faden mit klarer Reihenfolge der Informationen
    - Keine komplizierten Nebensatzkonstruktionen oder mehrfach verschachtelte Sätze
    - Wichtige Informationen sollen möglichst vollständig erhalten bleiben und nur so weit wie für die Zielgruppe nötig gekürzt/umschrieben werden
    """,
    "3": """
    Zielgruppe: Kinder und Jugendliche von 11 bis 14 Jahren
    Merkmale:
    - Klar strukturierte, aber etwas komplexere Sätze sind möglich
    - Wortschatz der Altersklasse entsprechend
    - Fachbegriffe mit knapper Erklärung
    - Zusammenhänge und Hintergründe dürfen etwas ausführlicher dargestellt werden, bleiben aber verständlich
    - Wichtige Informationen sollen möglichst vollständig erhalten bleiben und nur so weit wie für die Zielgruppe nötig gekürzt/umschrieben werden
    """
}


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def simplify_GPT(age_group, input_text):
    # age_group entweder "1" (5-8), "2" (8-11) oder "3" (11-14)
    zielgruppe = AGE_GROUP_DESCRIPTIONS[age_group]

    system_prompt = f"""Du vereinfachst deutsche Lexikontexte für Kinder.
    {zielgruppe}

    Gib NUR den vereinfachten Text zurück, ohne Einleitung, ohne Kommentar,
    ohne Anführungszeichen drumherum."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ],
        #temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def main():
    output_dir = Path(OUTPUT_DIR)
    results_path = output_dir / "results_GPT.jsonl"

    source_texts = load_jsonl(SOURCE_TEXT_PATH)
    # source_texts = source_texts[11:]

    for text in source_texts:
        source_text = text["text"]
        append_jsonl(results_path, source_text)

        result = simplify_GPT("1", source_text)
        append_jsonl(results_path, result)
        print(f"done 1. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = simplify_GPT("2", source_text)
        append_jsonl(results_path, result)
        print(f"done 2. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = simplify_GPT("3", source_text)
        append_jsonl(results_path, result)
        print(f"done 3. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()