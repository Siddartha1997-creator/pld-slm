import json
import inspect
from datasets import Dataset
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    Trainer,
    TrainingArguments,
)

# 1. Load BIO-tagged data: tokens + tags
def load_ner_jsonl(path):
    all_tokens, all_tags = [], []
    for line in open(path):
        obj = json.loads(line)
        all_tokens.append(obj["tokens"])
        all_tags.append(obj["tags"])
    return all_tokens, all_tags

tokens_list, tags_list = load_ner_jsonl("../data/entities/train.jsonl")

# 2. Build tag2id / id2tag from dataset
unique_tags = sorted({tag for seq in tags_list for tag in seq})
tag2id = {t: i for i, t in enumerate(unique_tags)}
id2tag = {i: t for t, i in tag2id.items()}

# 3. Wrap in a HuggingFace Dataset and split into train/val
full_dataset = Dataset.from_dict(
    {"tokens": tokens_list, "tags": [[tag2id[t] for t in seq] for seq in tags_list]}
)

split = full_dataset.train_test_split(test_size=0.15, seed=42)
train_dataset_raw = split["train"]
eval_dataset_raw = split["test"]

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_and_align_labels(batch):
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=True,
    )
    all_labels = []
    for i in range(len(batch["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        labels = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)  # ignore in loss
            else:
                labels.append(batch["tags"][i][word_id])
        all_labels.append(labels)
    tokenized["labels"] = all_labels
    return tokenized

dataset = train_dataset_raw.map(tokenize_and_align_labels, batched=True)
eval_dataset = eval_dataset_raw.map(tokenize_and_align_labels, batched=True)

# 4. Model + training
model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(tag2id),
    id2label=id2tag,
    label2id=tag2id,
)

_ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

_candidate_args = {
    "output_dir": "models/ner",
    "per_device_train_batch_size": 8,
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
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
trainer.save_model("models/ner")