from datasets import load_from_disk
from typing import List, Dict, Any
import random
from config import DATA_DIR


class QADataset:
    """handler for HotpotQA dataset."""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.dataset = None
        self.train = None
        self.validation = None
    
    def load(self):
        """load dataset from disk."""

        self.dataset = load_from_disk(self.data_dir)
        self.train = self.dataset["train"]
        self.validation = self.dataset["validation"]
        return self
    
    def sample_batch(self, batch_size: int, split: str = "train") -> List[Dict[str, Any]]:
        """sample random batch from dataset."""

        data = self.train if split == "train" else self.validation
        if data is None or len(data) == 0:
            return []
        sample_size = min(batch_size, len(data))
        indices = random.sample(range(len(data)), sample_size)
        return [data[i] for i in indices]
    
    def get_split(self, split: str = "train"):
        """get full split."""

        return self.train if split == "train" else self.validation
    
    def get_example(self, idx: int, split: str = "train") -> Dict[str, Any]:
        """get single example by index."""

        data = self.train if split == "train" else self.validation
        return data[idx]
    
    def get_ground_truth(self, example: Dict[str, Any]) -> str:
        """extract ground truth answer from example."""

        return example["answer"]
    
    def get_question(self, example: Dict[str, Any]) -> str:
        """extract question from example."""

        return example["question"]
    
    def get_context(self, example: Dict[str, Any]) -> str:
        """extract and format context from example."""

        contexts = []
        for ctx in example["context"]["sentences"]:
            contexts.extend(ctx)
        return " ".join(contexts)
    
    def __len__(self):
        return len(self.train) if self.train else 0

