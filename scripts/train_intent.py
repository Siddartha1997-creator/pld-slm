import json
import inspect
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer, TrainingArguments
)

with open("../data/intent_schema.json") as f:
    label_map = json.load(f)

def load_jsonl(path):
    texts, labels = [], []
    for line in open(path):
        obj = json.loads(line)
        texts.append(obj["text"])
        labels.append(label_map[obj["intent"]])
    return Dataset.from_dict({"text": texts, "label": labels})

train_ds = load_jsonl("../data/intents/train_balanced.jsonl")
val_ds   = load_jsonl("../data/intents/val.jsonl")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_map)
)

_ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

_candidate_args = {
    "output_dir": "models/intent_classifier",
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 5,
    "logging_steps": 1,
    "weight_decay": 0.01,
}

# Mixed precision is only valid on CUDA in most setups.
_use_fp16 = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
if "fp16" in _ta_params:
    _candidate_args["fp16"] = _use_fp16

# Transformers versions differ on these argument names.
if "evaluation_strategy" in _ta_params:
    _candidate_args["evaluation_strategy"] = "epoch"
elif "evaluate_during_training" in _ta_params:
    _candidate_args["evaluate_during_training"] = True

if "save_strategy" in _ta_params:
    _candidate_args["save_strategy"] = "epoch"

# Older versions used train_batch_size/eval_batch_size instead.
if "per_device_train_batch_size" not in _ta_params and "train_batch_size" in _ta_params:
    _candidate_args["train_batch_size"] = _candidate_args.pop("per_device_train_batch_size")
if "per_device_eval_batch_size" not in _ta_params and "eval_batch_size" in _ta_params:
    _candidate_args["eval_batch_size"] = _candidate_args.pop("per_device_eval_batch_size")

args = TrainingArguments(**{k: v for k, v in _candidate_args.items() if k in _ta_params})

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()
trainer.save_model("models/intent_classifier")