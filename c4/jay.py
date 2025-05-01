from dataset_builder import build_dataset
import json

# Generate samples
dataset = build_dataset(n_samples=4000)

# Save to JSON file
with open("connect_four_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

