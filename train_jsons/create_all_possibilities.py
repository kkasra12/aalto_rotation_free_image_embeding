import json
from pathlib import Path

models = ["resnet18", "resnet50", "simple"]
distances = ["euclidean", "cosine"]

output_dir = Path(__file__).parent

for model in models:
    for distance in distances:
        config = {
            "cnn": model,
            "loss-margin": 1.0,
            "distance": distance,
            "checkpoint_path": f"checkpoints/{model}_{distance}",
            "train": "/home/users/keskandarizanjani/datasets/tanks_and_temples",
            "m": 150,
        }
        out_path = output_dir / f"config_{model}_{distance}.json"
        print(f"Configuring model {model} with distance {distance}")
        out_path.write_text(json.dumps(config, indent=2))
