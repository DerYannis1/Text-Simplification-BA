from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from pathlib import Path
from datetime import datetime


SOURCE_TEXT_PATH = "./human_evaluation/source_text.jsonl"
OUTPUT_DIR = "./human_evaluation"
#HF_TOKEN hier festlegen, aber lieber nicht auf GitHub pushen


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

def simplify_German4All(level, input_text):
    #level entweder 1, 2 oder 3
    model_name = "tum-nlp/single_model_xl"
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", token=HF_TOKEN)
    model = PeftModel.from_pretrained(base_model, model_name, token=HF_TOKEN).to(base_model.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    description = """Verwandle den folgenden Text in die Zielsprachebene gemäß den untenstehenden Beschreibungen.
    KOMPLEXITÄTSBESCHREIBUNGEN
    Level 1 : Leichte Sprache:
    Zielgruppe: Zielgruppe: Personen mit Leseschwierigkeiten, inklusive Menschen mit Lernbehinderungen und solche, die erst kürzlich begonnen haben, Deutsch zu lernen.
    Merkmale: Sehr kurze Sätze, nur kurze und häufig verwendete Wörter, direkte Ansprache. Vermeidung von Abkürzungen, Metaphern oder Ironie.
    Beispiele: Einfache Anleitungen, barrierefreie Webseiten.

    Level 2 : Einfaches Deutsch für Anfänger:
    Zielgruppe: Nicht-Muttersprachler mit grundlegenden Deutschkenntnissen.
    Merkmale: Einfache Satzstrukturen, Grundwortschatz, klarer Fokus auf wichtige Informationen, Vermeidung kulturspezifischer Ausdrücke.
    Beispiele:  Sprachlernmaterialien, einführende Webtexte.

    Level 3 : Gebräuchliche Standardsprache:
    Zielgruppe: Öffentlichkeit mit unterschiedlichem Bildungsniveau.
    Merkmale: Klare, strukturierte Sätze, Verständlichkeit steht im Vordergrund, Vermeidung von Fachjargon.
    Beispiele: Breit gefächerte Nachrichtenportale, Blogs.
    """

    prefix = f"\n Paraphrasiere den folgenden Text auf Level {level}.\n Text: "
    input_full = [description + prefix.format(level=level) + input_text]
    model_inputs = tokenizer(input_full, max_length=512, truncation=True, padding="max_length",
                            return_tensors="pt").to(model.device)
    eval_preds = model.generate(**model_inputs, do_sample=True, top_p=0.9, temperature=0.1,
                            max_new_tokens=256)
    output_preds = tokenizer.batch_decode(eval_preds, skip_special_tokens=True)[0]
    
    return output_preds

def main():

    output_dir = Path(OUTPUT_DIR)
    results_path = output_dir / "results_G4All.jsonl"

    source_texts = load_jsonl(SOURCE_TEXT_PATH)
    #source_texts = source_texts[11:]

    for text in source_texts:
        source_text = text["text"]
        append_jsonl(results_path, source_text)
        result = simplify_German4All("1", source_text)
        append_jsonl(results_path, result)
        print(f"done 1. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = simplify_German4All("2", source_text)
        append_jsonl(results_path, result)
        print(f"done 2. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = simplify_German4All("3", source_text)
        append_jsonl(results_path, result)
        print(f"done 3. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()