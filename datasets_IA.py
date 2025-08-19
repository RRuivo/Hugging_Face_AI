# from datasets import load_dataset

# ds = load_dataset("facebook/natural_reasoning", split="train")


# print(ds[0]["question"])

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/Anthropic/EconomicIndex/release_2025_03_27/automation_vs_augmentation_by_task.csv")
print(df)