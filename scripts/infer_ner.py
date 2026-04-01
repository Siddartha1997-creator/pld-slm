import sys
import json
import argparse
from typing import List, Tuple, Dict

import torch
import requests
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, DistilBertForSequenceClassification


# -----------------------------
# Model loading
# -----------------------------
def load_ner_model(model_dir: str):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def load_intent_model(model_dir: str):
    with open("../data/intent_schema.json") as f:
        label_map = json.load(f)
    id2intent = {v: k for k, v in label_map.items()}
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model, id2intent


# -----------------------------
# NER prediction
# -----------------------------
def predict_ner(
    text: str,
    tokenizer: DistilBertTokenizerFast,
    model: DistilBertForTokenClassification,
) -> List[Tuple[str, str]]:
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        is_split_into_words=False,
    )

    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(dim=-1)[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    id2label = model.config.id2label

    results = []
    for token, label_id in zip(tokens, predictions):
        if token in ("[CLS]", "[SEP]"):
            continue
        label = id2label[label_id]
        results.append((token, label))

    return results


# -----------------------------
# Subword merge
# -----------------------------
def merge_subwords(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    merged = []

    for token, tag in pairs:
        if token.startswith("##"):
            prev_token, prev_tag = merged[-1]
            merged[-1] = (prev_token + token[2:], prev_tag)
        else:
            merged.append((token, tag))

    return merged


# -----------------------------
# BIO → entities
# -----------------------------
def extract_entities(pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    entities = {}
    current_type = None
    current_tokens = []

    for token, tag in pairs:
        if tag.startswith("B-"):
            if current_type:
                entities.setdefault(current_type, []).append(" ".join(current_tokens))
            current_type = tag[2:]
            current_tokens = [token]

        elif tag.startswith("I-") and current_type == tag[2:]:
            current_tokens.append(token)

        else:
            if current_type:
                entities.setdefault(current_type, []).append(" ".join(current_tokens))
                current_type = None
                current_tokens = []

    if current_type:
        entities.setdefault(current_type, []).append(" ".join(current_tokens))

    return entities


# -----------------------------
# Intent (model-based)
# -----------------------------
def predict_intent(
    text: str,
    tokenizer: DistilBertTokenizerFast,
    model: DistilBertForSequenceClassification,
    id2intent: Dict[int, str],
) -> str:
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**encoding).logits
    label_id = logits.argmax(dim=-1).item()
    return id2intent[label_id]


# -----------------------------
# Send to ESP32 (Option A)
# -----------------------------
def send_to_esp32(command: dict, url: str):
    try:
        r = requests.post(url, json=command, timeout=1)
        print(f"[ESP32] Sent → status {r.status_code}")
    except Exception as e:
        print(f"[ESP32] ERROR: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*", help="Input command")
    parser.add_argument("--send", action="store_true", help="Send command to ESP32")
    parser.add_argument("--esp32-url", default="http://esp32.local/command")
    args = parser.parse_args()

    text = " ".join(args.text) if args.text else input("Enter text: ").strip()

    ner_tokenizer, ner_model = load_ner_model("models/ner")
    intent_tokenizer, intent_model, id2intent = load_intent_model("models/intent_classifier")

    raw_pairs = predict_ner(text, ner_tokenizer, ner_model)
    merged_pairs = merge_subwords(raw_pairs)
    entities = extract_entities(merged_pairs)
    intent = predict_intent(text, intent_tokenizer, intent_model, id2intent)

    command = {
        "intent": intent,
        "device": entities.get("DEVICE", [None])[0],
        "location": entities.get("LOCATION", [None])[0],
        "value": entities.get("VALUE", [None])[0],
    }

    print("\n--- TOKENS ---")
    for t, tag in merged_pairs:
        print(f"{t:<12} {tag}")

    print("\n--- COMMAND ---")
    print(command)

    if args.send:
        send_to_esp32(command, args.esp32_url)


if __name__ == "__main__":
    main()