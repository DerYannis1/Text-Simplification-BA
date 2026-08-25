import argparse
import json
import random
from collections import defaultdict

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

LEVEL_TOKENS = ["<KLEXIKON>", "<MINIKLEXIKON>"]

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def tensorboard_available():
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        return True
    except ImportError:
        try:
            import tensorboardX  # noqa: F401
            return True
        except ImportError:
            return False


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def register_control_tokens(tokenizer, model):
    # <KLEXIKON>/<MINIKLEXIKON> sind mT5 nicht bekannt und wuerden sonst in
    # mehrere Sub-Word-Stuecke zerlegt statt ein eigenes Embedding zu bekommen.
    added = tokenizer.add_special_tokens({"additional_special_tokens": LEVEL_TOKENS})
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))


def preprocess_fn(tokenizer, max_source_len, max_target_len):
    def preprocess(batch):
        inputs = tokenizer(batch["input_text"], max_length=max_source_len, truncation=True)
        labels = tokenizer(text_target=batch["target_text"], max_length=max_target_len, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs
    return preprocess


def rouge_scores(preds, refs):
    r1, r2, rl = [], [], []
    for pred, ref in zip(preds, refs):
        if not ref:
            continue
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1": np.mean(r1) if r1 else 0.0,
        "rouge2": np.mean(r2) if r2 else 0.0,
        "rougeL": np.mean(rl) if rl else 0.0,
    }


def copy_rate(sources, preds):
    # Anteil der Vorhersagen, die groesstenteils aus Woertern der Quelle
    # bestehen -- soll erkennen, ob das Modell nur kopiert statt vereinfacht.
    copies = 0
    for src, pred in zip(sources, preds):
        src_words = set(src.lower().split())
        pred_words = set(pred.lower().split())
        if pred_words and len(pred_words & src_words) / len(pred_words) > 0.9:
            copies += 1
    return copies / max(len(preds), 1)


def d_sari_score(sources, preds, refs):
    scores = []
    for src, pred, ref in zip(sources, preds, refs):
        if not src.strip() or not pred.strip() or not ref.strip():
            continue
        final_score, _, _, _ = d_sari_sent(src, pred, [ref])
        scores.append(final_score)
    return np.mean(scores) if scores else 0.0


def compute_metrics_fn(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        metrics = rouge_scores(decoded_preds, decoded_labels)

        # Copy-Rate + D-SARI brauchen den Quelltext -- nur vorhanden, wenn
        # include_for_metrics=["inputs"] gesetzt ist.
        raw_inputs = getattr(eval_pred, "inputs", None)
        if raw_inputs is not None:
            inputs_arr = raw_inputs[0] if isinstance(raw_inputs, tuple) else raw_inputs
            inputs_arr = np.where(inputs_arr != -100, inputs_arr, tokenizer.pad_token_id)
            decoded_inputs = tokenizer.batch_decode(inputs_arr, skip_special_tokens=True)

            metrics["copy_rate"] = copy_rate(decoded_inputs, decoded_preds)
            metrics["d_sari"] = d_sari_score(decoded_inputs, decoded_preds, decoded_labels)

        return metrics
    return compute_metrics


def generate_batch(model, tokenizer, texts, args, batch_size=8):
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, max_length=args.max_source_length, truncation=True,
                             padding=True, return_tensors="pt").to(device)
            gen = model.generate(
                **enc,
                max_length=args.generation_max_length,
                num_beams=args.generation_num_beams,
            )
            outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def evaluate_by_level(model, tokenizer, val_records, args):
    # Getrennte Metriken pro Niveau, damit ein schwaches Niveau nicht im
    # gemittelten Gesamtwert untergeht.
    by_level = defaultdict(list)
    for r in val_records:
        by_level[r["level"]].append(r)

    results = {}
    for level, records in by_level.items():
        inputs = [r["input_text"] for r in records]
        refs = [r["target_text"] for r in records]
        preds = generate_batch(model, tokenizer, inputs, args, batch_size=args.batch_size)
        results[level] = rouge_scores(preds, refs)
        results[level]["n"] = len(records)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default="dataset_mt5/train.jsonl")
    p.add_argument("--val-file", default="dataset_mt5/val.jsonl")
    p.add_argument("--model-name", default="google/mt5-base")
    p.add_argument("--output-dir", default="./mt5-simplify-de")

    p.add_argument("--max-source-length", type=int, default=1024)
    p.add_argument("--max-target-length", type=int, default=512)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--num-train-epochs", type=float, default=20)
    p.add_argument("--optimizer", choices=["adamw_torch", "adafactor"], default="adamw_torch")

    p.add_argument("--generation-max-length", type=int, default=512)
    p.add_argument("--generation-num-beams", type=int, default=4)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)

    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--early-stopping-patience", type=int, default=4)

    p.add_argument("--report-to", choices=["tensorboard", "none"], default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)
    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    register_control_tokens(tokenizer, model)

    preprocess = preprocess_fn(tokenizer, args.max_source_length, args.max_target_length)
    train_ds_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_ds_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    report_to = [args.report_to] if args.report_to != "none" else []
    if "tensorboard" in report_to and not tensorboard_available():
        print("TensorBoard nicht installiert (pip install tensorboard) -- Logging wird deaktiviert.")
        report_to = []

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        optim=args.optimizer,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        bf16=True,
        report_to=report_to,
        seed=args.seed,
        include_for_metrics=["inputs"],
    )

    # Repetition-Kontrolle greift auch bei der Generierung waehrend Eval/Inferenz.
    model.generation_config.max_length = args.generation_max_length
    model.generation_config.num_beams = args.generation_num_beams
    model.generation_config.no_repeat_ngram_size = args.no_repeat_ngram_size
    model.generation_config.repetition_penalty = args.repetition_penalty

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_tok,
        eval_dataset=val_ds_tok,
        data_collator=collator,
        compute_metrics=compute_metrics_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Finale Validation-Metriken:")
    print(trainer.evaluate())

    print("Metriken pro Niveau:")
    print(evaluate_by_level(trainer.model, tokenizer, val_records, args))


if __name__ == "__main__":
    main()