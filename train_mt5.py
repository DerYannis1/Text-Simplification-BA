#!/usr/bin/env python3
"""
Fine-Tuning von mT5-base fuer kontrollierbare deutsche Textvereinfachung
(Wikipedia -> Klexikon / MiniKlexikon), auf Basis von train/val.jsonl aus
prepare_training_data.py.

Erwartetes Datenformat pro Zeile (JSON):
    {"id": ..., "level": "Klexikon"|"MiniKlexikon",
     "input_text": "vereinfache <LEVEL>: <quelltext>",
     "target_text": "<zieltext>",
     "grounding_score": float, "low_grounding_flag": bool}

Beispielaufruf:
    python train_mt5.py --train-file dataset/train.jsonl --val-file dataset/val.jsonl \
        --output-dir ./mt5-simplify-de --per-device-train-batch-size 4 \
        --gradient-accumulation-steps 8 --num-train-epochs 6

Hinweis zur transformers-Version: Die Parameter unten (insbesondere
eval_strategy statt evaluation_strategy, include_for_metrics statt
include_inputs_for_metrics, warmup_steps statt warmup_ratio) wurden gegen
transformers 5.x geprueft. Bei einer aelteren installierten Version kann es
sein, dass sich einzelne Argumentnamen unterscheiden -- die Fehlermeldung
nennt dann den erwarteten Namen.
"""

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from d_sari import d_sari_sent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

# Fuer D-SARI wird der reine Quelltext benoetigt (ohne "vereinfache <LEVEL>: "-
# Praefix aus format_t5_example() in prepare_training_data.py).
_PREFIX_RE = re.compile(r"^vereinfache\s+<[^>]+>:\s*", re.IGNORECASE)

# NEU: Die Niveau-Kontrolltoken aus prepare_training_data.py's format_t5_example().
# mT5s SentencePiece-Vokabular kennt diese Strings nicht -- ohne explizite
# Registrierung werden sie in mehrere Sub-Word-Fragmente zerlegt statt als
# EIN dediziertes, trainierbares Embedding behandelt zu werden. Das
# verwaessert das Steuersignal erheblich (siehe register_control_tokens()).
LEVEL_CONTROL_TOKENS = ["<KLEXIKON>", "<MINIKLEXIKON>"]


def _tensorboard_available():
    """Eigene Pruefung statt einer internen transformers-Utility-Funktion --
    is_tensorboard_available() ist in aktuellen transformers-Versionen nicht
    mehr stabil unter transformers.utils exponiert (gegen 5.15.0 geprueft).
    Bildet exakt nach, was der interne TensorBoardCallback selbst prueft."""
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        return True
    except ImportError:
        try:
            import tensorboardX  # noqa: F401
            return True
        except ImportError:
            return False


def register_control_tokens(tokenizer, model):
    """Registriert die Niveau-Kontrolltoken als atomare Special Tokens mit
    eigenem, frisch initialisiertem Embedding, statt sie als Sub-Word-
    Fragmente zu belassen. Muss VOR der Tokenisierung des Datensatzes
    aufgerufen werden."""
    added = tokenizer.add_special_tokens({"additional_special_tokens": LEVEL_CONTROL_TOKENS})
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(
            "Kontroll-Token als dedizierte Special Tokens registriert (%d neu, "
            "Embedding-Matrix auf %d Zeilen vergroessert): %s",
            added, len(tokenizer), LEVEL_CONTROL_TOKENS,
        )
    else:
        logger.info("Kontroll-Token waren bereits registriert.")


# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_hf_dataset(records, exclude_low_grounding=False):
    if exclude_low_grounding:
        before = len(records)
        records = [r for r in records if not r.get("low_grounding_flag", False)]
        logger.info(
            "Low-grounding-Beispiele ausgeschlossen: %d von %d entfernt.",
            before - len(records), before,
        )
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Tokenisierung
# ---------------------------------------------------------------------------

def make_preprocess_fn(tokenizer, max_source_length, max_target_length):
    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


# ---------------------------------------------------------------------------
# Metriken / Diagnostik
# ---------------------------------------------------------------------------
# WICHTIG: mT5-Fine-Tuning auf Vereinfachungsdaten neigt manchmal dazu, auf
# sehr kurze, generische Saetze zu kollabieren (siehe z.B. das Verhalten der
# "mt5-SGC"-Baseline im German4All-Paper, Anschuetz et al. 2025, INLG). Daher
# werden hier zusaetzlich zu ROUGE einfache Laengen-/Kollaps-Diagnosen
# mitgeloggt -- nicht nur die reine Qualitaetsmetrik.

def make_compute_metrics_fn(tokenizer):
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        r1, r2, rl = [], [], []
        for pred, ref in zip(decoded_preds, decoded_labels):
            if not ref:
                continue
            scores = _scorer.score(ref, pred)
            r1.append(scores["rouge1"].fmeasure)
            r2.append(scores["rouge2"].fmeasure)
            rl.append(scores["rougeL"].fmeasure)

        pred_lengths = [len(p.split()) for p in decoded_preds]
        target_lengths = [len(l.split()) for l in decoded_labels]
        short_rate = sum(1 for n in pred_lengths if n < 5) / max(len(pred_lengths), 1)

        metrics = {
            "rouge1": float(np.mean(r1)) if r1 else 0.0,
            "rouge2": float(np.mean(r2)) if r2 else 0.0,
            "rougeL": float(np.mean(rl)) if rl else 0.0,
            "avg_pred_words": float(np.mean(pred_lengths)) if pred_lengths else 0.0,
            "avg_target_words": float(np.mean(target_lengths)) if target_lengths else 0.0,
            "short_output_rate": short_rate,
        }

        # Copy-Rate + D-SARI: beide brauchen den rohen Quelltext, nur
        # berechenbar, wenn Inputs mitgeliefert wurden
        # (include_for_metrics=["inputs"]); sonst wird still uebersprungen,
        # um Versions-Kompatibilitaetsprobleme nicht zum Hard-Fail zu machen.
        raw_inputs = getattr(eval_pred, "inputs", None)
        if raw_inputs is not None:
            try:
                inputs_arr = raw_inputs[0] if isinstance(raw_inputs, tuple) else raw_inputs
                inputs_arr = np.where(inputs_arr != -100, inputs_arr, tokenizer.pad_token_id)
                decoded_inputs = tokenizer.batch_decode(inputs_arr, skip_special_tokens=True)
                copies = 0
                for src, pred in zip(decoded_inputs, decoded_preds):
                    src_words = set(src.lower().split())
                    pred_words = set(pred.lower().split())
                    if pred_words and len(pred_words & src_words) / len(pred_words) > 0.9:
                        copies += 1
                metrics["copy_rate"] = copies / max(len(decoded_preds), 1)
            except Exception as e:
                logger.warning("Copy-Rate-Diagnose uebersprungen: %s", e)

            # NEU: D-SARI (Sun, Jin & Wan, 2021) -- fuer DOKUMENT-Ebene
            # entwickelte SARI-Variante mit Laengen-/Satzanzahl-Straftermen,
            # geeigneter als das normale (satzweise konzipierte) SARI aus
            # evaluate_mt5.py fuer eure mehrsaetzigen Klexikon/MiniKlexikon-
            # Ausgaben. Braucht den quellenrohen Text (Praefix entfernt).
            try:
                sources = [_PREFIX_RE.sub("", s, count=1) for s in decoded_inputs]
                d_sari_scores = []
                for src, pred, ref in zip(sources, decoded_preds, decoded_labels):
                    if not src.strip() or not pred.strip() or not ref.strip():
                        continue
                    final_score, _, _, _ = d_sari_sent(src, pred, [ref])
                    d_sari_scores.append(final_score)
                metrics["d_sari"] = float(np.mean(d_sari_scores)) if d_sari_scores else 0.0
            except Exception as e:
                logger.warning("D-SARI-Berechnung uebersprungen: %s", e)

        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# Generierung fuer Diagnose-Auswertungen nach dem Training
# ---------------------------------------------------------------------------

def _generate_batch(model, tokenizer, texts, args, batch_size=8):
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tokenizer(
                batch,
                max_length=args.max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            gen = model.generate(
                **enc,
                max_length=args.generation_max_length,
                min_length=args.generation_min_length,
                num_beams=args.generation_num_beams,
            )
            outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def evaluate_by_level(model, tokenizer, val_records, args):
    """Separate ROUGE-/Laengen-Metriken pro Niveau -- prueft, ob das Modell
    fuer Klexikon UND MiniKlexikon gleichermassen gut funktioniert, statt nur
    einen gemittelten Gesamtwert zu sehen, der ein schwaches Niveau
    verschleiern koennte."""
    by_level = defaultdict(list)
    for r in val_records:
        by_level[r["level"]].append(r)

    results = {}
    for level, records in by_level.items():
        inputs = [r["input_text"] for r in records]
        refs = [r["target_text"] for r in records]
        preds = _generate_batch(model, tokenizer, inputs, args, batch_size=args.per_device_eval_batch_size)

        r1, r2, rl = [], [], []
        for pred, ref in zip(preds, refs):
            scores = _scorer.score(ref, pred)
            r1.append(scores["rouge1"].fmeasure)
            r2.append(scores["rouge2"].fmeasure)
            rl.append(scores["rougeL"].fmeasure)

        pred_lengths = [len(p.split()) for p in preds]
        results[level] = {
            "n": len(records),
            "rouge1": float(np.mean(r1)) if r1 else 0.0,
            "rouge2": float(np.mean(r2)) if r2 else 0.0,
            "rougeL": float(np.mean(rl)) if rl else 0.0,
            "avg_pred_words": float(np.mean(pred_lengths)) if pred_lengths else 0.0,
        }
    return results


def preview_control_token_effect(model, tokenizer, val_records, args):
    """Sanity-Check: fuer denselben Artikel (gleiche id) beide Niveaus
    generieren und nebeneinander ausgeben. Wenn das Kontroll-Token wirkt,
    sollten sich die beiden Ausgaben spuerbar unterscheiden -- nicht nur
    zufaellig leicht variieren oder identisch sein. Bei langen Quelltexten
    (bis ~1500 Woerter) ist das kein Selbstlaeufer: ein einzelnes Token kann
    im Encoder-Input "untergehen"."""
    by_id = defaultdict(dict)
    for r in val_records:
        by_id[r["id"]][r["level"]] = r

    pairs = [v for v in by_id.values() if "Klexikon" in v and "MiniKlexikon" in v]
    random.shuffle(pairs)
    pairs = pairs[: args.num_sample_previews]

    if not pairs:
        logger.warning("Keine id mit beiden Niveaus im Val-Set gefunden -- Kontroll-Token-Check uebersprungen.")
        return

    logger.info("=" * 70)
    logger.info("KONTROLL-TOKEN-CHECK: gleicher Artikel, beide Niveaus")
    logger.info("=" * 70)
    for pair in pairs:
        klex_in = pair["Klexikon"]["input_text"]
        mini_in = pair["MiniKlexikon"]["input_text"]
        preds = _generate_batch(model, tokenizer, [klex_in, mini_in], args, batch_size=2)
        logger.info("--- %s ---", pair["Klexikon"]["id"])
        logger.info("Klexikon-Output:     %s", preds[0][:300])
        logger.info("MiniKlexikon-Output: %s", preds[1][:300])
        if preds[0].strip() == preds[1].strip():
            logger.warning(
                "  ACHTUNG: identischer Output fuer beide Niveaus -- "
                "Kontroll-Token wird moeglicherweise ignoriert!"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-file", default="dataset/train.jsonl")
    p.add_argument("--val-file", default="dataset/val.jsonl")
    p.add_argument("--model-name", default="google/mt5-base")
    p.add_argument("--output-dir", default="./mt5-simplify-de")

    p.add_argument("--max-source-length", type=int, default=1024,
                    help="Sollte mit --max-wiki-words aus dem Corpus-Builder zusammenpassen "
                         "(1500 Woerter Deutsch ~ 2000+ Subword-Tokens); laengere Quellen werden "
                         "hier hart abgeschnitten.")
    p.add_argument("--max-target-length", type=int, default=512)

    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--per-device-eval-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8,
                    help="Effektive Batch-Groesse = per_device_train_batch_size * dieser Wert.")

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--num-train-epochs", type=float, default=20,
                    help="6 Epochen (450 Steps bei 2416 Beispielen) waren in Tests zu wenig, um "
                         "das Niveau-Kontrolltoken zuverlaessig zu lernen -- 20 als neuer Default, "
                         "aber eval_rougeL/avg_pred_words im Log beobachten und ggf. anpassen.")
    p.add_argument("--warmup-ratio", type=float, default=0.06,
                    help="Wird intern in warmup_steps umgerechnet (warmup_ratio ist in aktuellen "
                         "transformers-Versionen fuer Seq2SeqTrainingArguments nicht mehr direkt "
                         "verfuegbar).")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing-factor", type=float, default=0.1,
                    help="Hilft gegen Kollaps auf generische kurze Outputs.")
    p.add_argument("--optimizer", choices=["adamw_torch", "adafactor"], default="adamw_torch",
                    help="adafactor ist die im T5-Paper vorgeschlagene, speichersparendere Wahl.")

    p.add_argument("--generation-max-length", type=int, default=512)
    p.add_argument("--generation-min-length", type=int, default=32,
                    help="Schutz gegen Ein-Satz-/Kollaps-Outputs bei Evaluation und Inferenz.")
    p.add_argument("--generation-num-beams", type=int, default=4)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3,
                    help="Verhindert Wiederholungs-Loops (z.B. denselben Satz mehrfach "
                         "hintereinander) -- ohne das neigt Beam Search bei einem noch "
                         "unsicheren Modell stark dazu.")
    p.add_argument("--repetition-penalty", type=float, default=1.3,
                    help="Zusaetzliche Bestrafung fuer bereits generierte Token; 1.0 = aus.")
    p.add_argument("--eval-num-beams", type=int, default=2,
                    help="Kleinere Beam-Zahl NUR fuer die (haeufigen) Trainings-Zwischen-"
                         "evaluationen, da Beam-Search bei langen Sequenzen sehr langsam ist "
                         "(~5 Min fuer 134 Beispiele bei num_beams=4). Fuer die finale Vorschau "
                         "wird weiterhin --generation-num-beams verwendet.")

    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16",
                    help="bf16 nur auf entsprechend unterstuetzter Hardware (z.B. neuere NVIDIA-GPUs); "
                         "sonst fp16 oder fp32 verwenden.")

    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--early-stopping-patience", type=int, default=4)

    p.add_argument("--exclude-low-grounding", action="store_true",
                    help="Ueberspringt Trainingsbeispiele mit low_grounding_flag=true aus "
                         "prepare_training_data.py.")
    p.add_argument("--exclude-low-grounding-val", action="store_true",
                    help="NEU: filtert zusaetzlich auch das Validation-Set. VORSICHT: die "
                         "Zwischenevaluation waehrend des Trainings (eval_rougeL usw.) misst dann "
                         "nur noch auf einer LEICHTEREN Teilmenge -- Werte sind nicht mehr direkt "
                         "vergleichbar mit Laeufen ohne dieses Flag (auch nicht mit fruehen Laeufen "
                         "in diesem Sweep!), und load_best_model_at_end/EarlyStopping waehlen den "
                         "Checkpoint dann nach einem Massstab, der die reale Verteilung nicht mehr "
                         "widerspiegelt. Default: aus, damit die Evaluation die reale Verteilung "
                         "zeigt.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--report-to", choices=["tensorboard", "none"], default="tensorboard",
                    help="TensorBoard-Logs landen unter {output_dir}/runs/... (automatisch von "
                         "transformers vergeben, in dieser Version nicht mehr ueber logging_dir "
                         "konfigurierbar). Fuer einen Hyperparameter-Sweep: 'tensorboard --logdir "
                         "<sweep_dir>' nach dem Sweep zeigt alle Laeufe nebeneinander, weil jeder "
                         "Lauf bereits in seinem eigenen, aussagekraeftig benannten --output-dir liegt.")
    p.add_argument("--num-sample-previews", type=int, default=5,
                    help="Anzahl Beispiel-Generierungen (beide Niveaus, gleicher Artikel) am Ende "
                         "zur manuellen Kontrolle, ob das Kontroll-Token wirkt.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    logger.info("Lade Daten ...")
    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)
    logger.info("train: %d Beispiele, val: %d Beispiele", len(train_records), len(val_records))

    train_ds = to_hf_dataset(train_records, exclude_low_grounding=args.exclude_low_grounding)
    if args.exclude_low_grounding_val:
        logger.warning(
            "--exclude-low-grounding-val aktiv: eval_rougeL & Co. messen jetzt auf einer "
            "leichteren Teilmenge des Val-Sets -- nicht direkt vergleichbar mit Laeufen ohne "
            "dieses Flag (z.B. anderen Konfigurationen im selben Hyperparameter-Sweep!)."
        )
    val_ds = to_hf_dataset(val_records, exclude_low_grounding=args.exclude_low_grounding_val)

    logger.info("Lade Tokenizer/Modell: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # NEU: tie_word_embeddings=False explizit setzen -- folgt der Warnung, die
    # transformers beim Laden von mT5-Checkpoints mit separat vorliegenden
    # shared.weight/lm_head.weight ausgibt ("... so we will NOT tie them.
    # You should update the config with tie_word_embeddings=False").
    # Ohne das kann es beim Neuladen des besten Checkpoints (load_best_model_at_end)
    # zu uneindeutigem Embedding-Handling kommen (siehe "missing keys" Warnung).
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, tie_word_embeddings=False)
    register_control_tokens(tokenizer, model)

    preprocess_fn = make_preprocess_fn(tokenizer, args.max_source_length, args.max_target_length)
    train_ds_tok = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds_tok = val_ds.map(preprocess_fn, batched=True, remove_columns=val_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    steps_per_epoch = max(
        len(train_ds_tok) // (args.per_device_train_batch_size * args.gradient_accumulation_steps), 1
    )
    total_steps = int(steps_per_epoch * args.num_train_epochs)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 0)
    logger.info("Geschaetzte Trainings-Steps: %d, davon Warmup: %d", total_steps, warmup_steps)

    report_to = [args.report_to] if args.report_to != "none" else []
    if "tensorboard" in report_to and not _tensorboard_available():
        logger.warning(
            "--report-to tensorboard angefordert, aber weder torch.utils.tensorboard noch "
            "tensorboardX verfuegbar (`pip install tensorboard`). Logging wird deaktiviert, "
            "damit das Training nicht mit einem RuntimeError abbricht."
        )
        report_to = []

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing_factor,
        optim=args.optimizer,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        # NEU: eval_num_beams statt generation_num_beams -- diese Zwischen-
        # Evaluationen laufen waehrend des Trainings sehr haeufig, daher hier
        # bewusst eine kleinere, schnellere Beam-Zahl. Die finale Vorschau
        # (evaluate_by_level/preview_control_token_effect) nutzt weiterhin
        # die hoehere --generation-num-beams.
        generation_num_beams=args.eval_num_beams,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=(args.precision == "fp16"),
        bf16=(args.precision == "bf16"),
        report_to=report_to,
        seed=args.seed,
        include_for_metrics=["inputs"],
    )

    # min_length/Repetition-Kontrolle fuer die Generation waehrend Evaluation/
    # Inferenz absichern. no_repeat_ngram_size + repetition_penalty verhindern
    # die Wiederholungs-Loops, die ohne diese Einstellungen auftreten koennen
    # (siehe Docstring oben).
    model.generation_config.min_length = args.generation_min_length
    model.generation_config.max_length = args.generation_max_length
    model.generation_config.num_beams = args.eval_num_beams
    model.generation_config.no_repeat_ngram_size = args.no_repeat_ngram_size
    model.generation_config.repetition_penalty = args.repetition_penalty

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_tok,
        eval_dataset=val_ds_tok,
        data_collator=collator,
        compute_metrics=make_compute_metrics_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    logger.info("Starte Training ...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Speichere finales (bestes) Modell nach %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Finale Validation-Metriken (gesamt):")
    final_metrics = trainer.evaluate()
    for k, v in final_metrics.items():
        logger.info("  %s: %s", k, v)

    logger.info("Metriken pro Niveau:")
    per_level_metrics = evaluate_by_level(trainer.model, tokenizer, val_records, args)
    for level, m in per_level_metrics.items():
        logger.info("  %s: %s", level, m)

    preview_control_token_effect(trainer.model, tokenizer, val_records, args)

    # NEU: strukturierte Zusammenfassung schreiben (Hyperparameter + finale
    # Metriken), damit ein Sweep-Skript (z.B. run_hyperparameter_sweep.py)
    # die Ergebnisse mehrerer Laeufe automatisiert vergleichen kann, ohne
    # den Log-Text parsen zu muessen.
    summary = {
        "output_dir": args.output_dir,
        "args": vars(args),
        "final_metrics": final_metrics,
        "per_level_metrics": per_level_metrics,
    }
    summary_path = Path(args.output_dir) / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Trainings-Zusammenfassung geschrieben nach %s", summary_path)


if __name__ == "__main__":
    main()