"""
Benötigte Schritte beim Fine Tuning:

1. Daten laden
2. Tokenizer um <KLEXIKON> / <MINIKLEXIKON> erweitern
3. Modell-Embeddings anpassen (resize_token_embeddings) für die neuen Tokens
4- Preprocessing, um passende Steuer-Token vor den Quelltext zu hängen
5. LoRA-Config mit modules_to_save für die Embedding-Layer
6. Trainings-Argumente festlegen und Training starten über Seq2SeqTrainer
7. Speichern & Inferenz mit wählbarem Ziel-Niveau

"""

from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "./dataset/train.jsonl",
    "validation": "./dataset/validation.jsonl",
    "test": "./dataset/test.jsonl",
})

#Debug: print(dataset)

# 2:

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "facebook/mbart-large-50"

tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)
control_tokens = ["<KLEXIKON>", "<MINIKLEXIKON>"]

token_added = tokenizer.add_special_tokens({
    "additional_special_tokens": control_tokens
})
#Debug: 
#print(f"Neue Tokens hinzugefügt: {token_added}")
#print(f"Neue Vokabelgröße: {len(tokenizer)}")


# 3:

import torch

old_vocab_size = model.get_input_embeddings().weight.shape[0]
#Debug: print(f"Alte Vokabelgröße (Modell): {old_vocab_size}")

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

with torch.no_grad():
    embeddings = model.get_input_embeddings()
    mean_embedding = embeddings.weight[:old_vocab_size].mean(dim=0)

    for token in control_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        embeddings.weight[token_id] = mean_embedding.clone()

#Debug:
# print(f"Neue Vokabelgröße : {model.get_input_embeddings().weight.shape[0]}")

# test_input = tokenizer("<MINIKLEXIKON> Dies ist ein Test.", return_tensors="pt")
# with torch.no_grad():
#     output = model(**test_input, decoder_input_ids=test_input["input_ids"])
# print("Forward-Pass erfolgreich, Output-Shape:", output.logits.shape)

# 4:

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 1024

tokenizer.src_lang = "de_DE"
tokenizer.tgt_lang = "de_DE"


def preprocess_function(examples):
    control_tokens = []
    sources_with_control = []

    for level in examples["level"]:
        if level == "Klexikon":
            control_tokens.append("<KLEXIKON>")
        else:
            control_tokens.append("<MINIKLEXIKON>")
    
    for ctrl, src in zip(control_tokens, examples["source"]):
        sources_with_control.append(f"{ctrl} {src}")

    model_inputs = tokenizer(
        sources_with_control,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

#Debug:
# print(tokenized_dataset)
# sample = tokenized_dataset["train"][0]

# decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
# decoded_label = tokenizer.decode(
#     [t for t in sample["labels"] if t != -100], 
#     skip_special_tokens=False
# )

#print("INPUT: ", decoded_input[:200], "...")
#print("LABEL: ", decoded_label[:200], "...")

#Debug for Layer names:
# linear_layer_names = set()
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         linear_layer_names.add(name.split(".")[-1])

# print("Gefundene lineare Layer-Typen:", linear_layer_names)

# embedding_layer_names = [name for name, _ in model.named_modules() if "embed" in name or "shared" in name]
# print("Embedding-bezogene Layer:", embedding_layer_names)

# 5:

from peft import LoraConfig, get_peft_model, TaskType

control_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in control_tokens]

#Debug:
# print(f"Token-IDs: {dict(zip(control_tokens, control_token_ids))}")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","v_proj"],
    trainable_token_indices={"model.shared": control_token_ids},
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

for name, param in model.named_parameters():
    if "token_adapter" in name or "trainable_tokens" in name:
        print(name, "-> shape:", param.shape, "requires_grad:", param.requires_grad)


#TRAINING

import numpy as np
import evaluate
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

#padding auf maximale länge im Batch
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=100,
    padding=True
)

rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    labels = np.where(label != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    #Whitesprache normalisieren
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=False #wären englische stemming Regeln
    )

    gen_lens = [len(tokenizer(pred)["input_ids"]) for pred in decoded_preds]
    result["gen_len"] = np.mean(gen_lens)

    return {k:round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir="./mbart-lora-TS",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better="True",

    learning_rate=1e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=10,

    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,

    fp16=True, 
    logging_steps=50,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=["validation"]
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)


model.generation_config.forced_bos_token_id = tokenizer.lang_code_to_id["de_DE"] #forced_bos_token_id (int, optional, defaults to model.config.forced_bos_token_id) — The id of the token to force as the first generated token after the decoder_start_token_id. Useful for multilingual models like mBART where the first generated token needs to be the target language token.
#trainer.train()


#Step 7 Speichern und INferenz mit wählbarem Ziel Niveau