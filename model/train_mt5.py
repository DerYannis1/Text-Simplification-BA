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


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def register_control_tokens(tokenizer, model):
    # <KLEXIKON>/<MINIKLEXIKON> token müssen als control Token hinzugefügt werden
    added = tokenizer.add_special_tokens({"additional_special_tokens": LEVEL_TOKENS})
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))


def preprocess_function(tokenizer, max_source_length, max_target_length):
    def preprocess(batch):
        inputs = tokenizer(batch["input_text"], max_length=max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["target_text"], max_length=max_target_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs
    return preprocess


def rouge_scores(preds, refs):
    r1, r2, rl = [], [], []
    for pred, reference in zip(preds, refs):
        if not reference:
            continue
        score = scorer.score(reference, pred)
        r1.append(score["rouge1"].fmeasure)
        r2.append(score["rouge2"].fmeasure)
        rl.append(score["rougeL"].fmeasure)
    return {
        "rouge1": np.mean(r1) if r1 else 0.0,
        "rouge2": np.mean(r2) if r2 else 0.0,
        "rougeL": np.mean(rl) if rl else 0.0,
    }


def copy_rate(sources, preds):
    # soll erkennen, ob das Modell nur kopiert statt vereinfacht
    copies = 0
    for source, pred in zip(sources, preds):
        source_words = set(source.lower().split())
        pred_words = set(pred.lower().split())
        if pred_words and len(pred_words & source_words) / len(pred_words) > 0.9:
            copies += 1
    return copies / max(len(preds), 1)


def d_sari_score(sources, preds, refs):
    scores = []
    for source, pred, reference in zip(sources, preds, refs):
        if not source.strip() or not pred.strip() or not reference.strip():
            continue
        final_score, _, _, _ = d_sari_sent(source, pred, [reference])
        scores.append(final_score)
    return np.mean(scores) if scores else 0.0


def compute_metrics_function(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        metrics = rouge_scores(decoded_preds, decoded_labels)

        raw_inputs = getattr(eval_pred, "inputs", None)
        if raw_inputs is not None:
            inputs_arr = raw_inputs[0] if isinstance(raw_inputs, tuple) else raw_inputs
            inputs_arr = np.where(inputs_arr != -100, inputs_arr, tokenizer.pad_token_id)
            decoded_inputs = tokenizer.batch_decode(inputs_arr, skip_special_tokens=True)

            metrics["copy_rate"] = copy_rate(decoded_inputs, decoded_preds)
            metrics["d_sari"] = d_sari_score(decoded_inputs, decoded_preds, decoded_labels)

        return metrics
    return compute_metrics


def generate_batch(model, tokenizer, texts, arguments, batch_size=8):
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(batch, max_length=arguments.max_source_length, truncation=True,
                                 padding=True, return_tensors="pt").to(device)
            generated = model.generate(
                **encoded,
                max_length=arguments.generation_max_length,
                num_beams=arguments.generation_num_beams,
            )
            outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def evaluate_by_level(model, tokenizer, val_records, arguments):
    by_level = defaultdict(list)
    for r in val_records:
        by_level[r["level"]].append(r)

    results = {}
    for level, records in by_level.items():
        inputs = [r["input_text"] for r in records]
        refs = [r["target_text"] for r in records]
        preds = generate_batch(model, tokenizer, inputs, arguments, batch_size=arguments.batch_size)
        results[level] = rouge_scores(preds, refs)
        results[level]["n"] = len(records)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="dataset_mt5/train.jsonl")
    parser.add_argument("--val-file", default="dataset_mt5/val.jsonl")
    parser.add_argument("--model-name", default="google/mt5-base")
    parser.add_argument("--output-dir", default="./mt5-simplify-de")

    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=512)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-train-epochs", type=float, default=20)
    parser.add_argument("--optimizer", choices=["adamw_torch", "adafactor"], default="adamw_torch")

    parser.add_argument("--generation-max-length", type=int, default=512)
    parser.add_argument("--generation-num-beams", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)

    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=4)

    parser.add_argument("--report-to", choices=["tensorboard", "none"], default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    arguments = parse_args()
    set_seed(arguments.seed)
    random.seed(arguments.seed)

    train_records = load_jsonl(arguments.train_file)
    val_records = load_jsonl(arguments.val_file)
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    tokenizer = AutoTokenizer.from_pretrained(arguments.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(arguments.model_name)
    register_control_tokens(tokenizer, model)

    preprocess = preprocess_function(tokenizer, arguments.max_source_length, arguments.max_target_length)
    train_dataset_tokenized = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    val_dataset_tokenized = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    report_to = [arguments.report_to] if arguments.report_to != "none" else []

    training_args = Seq2SeqTrainingArguments(
        output_dir=arguments.output_dir,
        per_device_train_batch_size=arguments.batch_size,
        per_device_eval_batch_size=arguments.batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        learning_rate=arguments.learning_rate,
        num_train_epochs=arguments.num_train_epochs,
        optim=arguments.optimizer,
        eval_strategy="steps",
        eval_steps=arguments.eval_steps,
        save_strategy="steps",
        save_steps=arguments.save_steps,
        save_total_limit=2,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=arguments.generation_max_length,
        generation_num_beams=arguments.generation_num_beams,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        bf16=True,
        report_to=report_to,
        seed=arguments.seed,
        include_for_metrics=["inputs"],
    )

    model.generation_config.max_length = arguments.generation_max_length
    model.generation_config.num_beams = arguments.generation_num_beams
    model.generation_config.no_repeat_ngram_size = arguments.no_repeat_ngram_size
    model.generation_config.repetition_penalty = arguments.repetition_penalty

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=val_dataset_tokenized,
        data_collator=collator,
        compute_metrics=compute_metrics_function(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=arguments.early_stopping_patience)],
    )

    trainer.train()

    trainer.save_model(arguments.output_dir)
    tokenizer.save_pretrained(arguments.output_dir)

    print("Finale Validation-Metriken:")
    print(trainer.evaluate())

    print("Metriken pro Niveau:")
    print(evaluate_by_level(trainer.model, tokenizer, val_records, arguments))


if __name__ == "__main__":
    main()