# PLD-SLM Project

A project for training and deploying intent classification and Named Entity Recognition (NER) models using DistilBERT.

## Project Structure

```
.
├── data/
│   ├── intent_schema.json           # Mapping of intents to class IDs
│   ├── entities/
│   │   └── train.jsonl              # Raw NER training data with BIO tags
│   ├── intents/
│   │   ├── train.jsonl              # Raw intent training data
│   │   ├── train_balanced.jsonl     # Balanced intent training data (generated)
│   │   ├── val.jsonl                # Intent validation data
│   │   └── noise/
│   │       └── noise.jsonl          # Noise data
│   
├── scripts/
│   ├── balance_intents.py           # Balance intent dataset by class
│   ├── train_intent.py              # Train intent classifier
│   ├── train_ner.py                 # Train NER model
│   ├── infer_ner.py                 # Run inference with trained models
│   └── models/                      # Trained model checkpoints (generated)
│
└── README.md                         # This file
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support recommended)
- Transformers library
- Datasets library

### Installation

```bash
pip install torch transformers datasets
```

## Running the Project

### Step 1: Balance Intent Dataset (Optional)

Balances the raw intent training data by oversampling minority classes:

```bash
cd scripts
python balance_intents.py
```

**Output:** `../data/intents/train_balanced.jsonl`

### Step 2: Train Intent Classifier

Trains a DistilBERT-based text classification model for intent detection:

```bash
python train_intent.py
```

**Output:** Trained model checkpoints in `models/intent_classifier/`

**Requirements:**
- `../data/intent_schema.json`
- `../data/intents/train_balanced.jsonl` (or use `train.jsonl` if skipping step 1)
- `../data/intents/val.jsonl`

### Step 3: Train NER Model

Trains a DistilBERT-based token classification model for Named Entity Recognition:

```bash
python train_ner.py
```

**Output:** Trained model checkpoints in `models/ner/`

**Requirements:**
- `../data/entities/train.jsonl`

### Step 4: Run Inference

Performs intent classification and NER on input text using trained models:

```bash
python infer_ner.py [options]
```

**Requires:**
- Trained intent classifier model
- Trained NER model

## Data Format

### Intent Data (train.jsonl)

```json
{"text": "show me the weather", "intent": "get_weather"}
{"text": "what's the temperature", "intent": "get_weather"}
```

### NER Data (entities/train.jsonl)

```json
{"tokens": ["book", "a", "flight", "to", "NYC"], "tags": ["O", "O", "O", "O", "LOC"]}
{"tokens": ["set", "a", "reminder"], "tags": ["O", "O", "O"]}
```

### Intent Schema (intent_schema.json)

```json
{
  "get_weather": 0,
  "set_reminder": 1,
  "book_flight": 2
}
```

## Models

- **Intent Classifier:** DistilBertForSequenceClassification
- **NER Model:** DistilBertForTokenClassification
- **Base Model:** distilbert-base-uncased

## Notes

- Models are trained on relatively small datasets; adjust hyperparameters in training scripts as needed
- GPU acceleration is recommended for training
- Mixed precision training (FP16) is enabled automatically if CUDA is available

## Development

All training and inference scripts are located in the `scripts/` directory. Modify the paths and hyperparameters directly in the scripts as needed.
