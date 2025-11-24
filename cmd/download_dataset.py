from datasets import load_dataset, load_from_disk
import os

data_dir = "data/hotpotqa"

if os.path.exists(data_dir):
    print(f"Dataset already downloaded at '{data_dir}'. Loading from disk...")
    dataset = load_from_disk(data_dir)
else:
    print(f"Dataset not found at '{data_dir}'. Downloading with Hugging Face Datasets...")
    dataset = load_dataset("hotpot_qa", "fullwiki")
    os.makedirs(data_dir, exist_ok=True)
    dataset.save_to_disk(data_dir)
    print(f"Dataset saved to '{data_dir}'")