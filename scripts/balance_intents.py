import json
from collections import Counter
import random

data = [json.loads(l) for l in open("../data/intents/train.jsonl")]
counts = Counter(d["intent"] for d in data)

max_count = max(counts.values())
balanced = []

for intent in counts:
    samples = [d for d in data if d["intent"] == intent]
    while len(samples) < max_count:
        samples.append(random.choice(samples))
    balanced.extend(samples)

random.shuffle(balanced)

with open("../data/intents/train_balanced.jsonl", "w") as f:
    for item in balanced:
        f.write(json.dumps(item) + "\n")

print("Balanced dataset written")