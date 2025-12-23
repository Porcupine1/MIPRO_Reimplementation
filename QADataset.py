from datasets import load_from_disk, concatenate_datasets
from typing import List, Dict, Any
import random
from config import DATA_DIR, MAX_EXAMPLES


class QADataset:
    """handler for HotpotQA dataset."""

    def __init__(self, data_dir: str = DATA_DIR, max_examples: int = MAX_EXAMPLES):
        self.data_dir = data_dir
        self.max_examples = max_examples
        self.dataset = None
        self.train = None
        self.validation = None

    def load(self):
        """load dataset from disk and split according to max_examples (80% train, 20% validation)."""

        self.dataset = load_from_disk(self.data_dir)
        original_train = self.dataset["train"]
        original_validation = self.dataset["validation"]

        # Combine train and validation datasets
        combined = concatenate_datasets([original_train, original_validation])

        # Sample max_examples total (or use all if max_examples is larger)
        if len(combined) > self.max_examples:
            combined = combined.shuffle(seed=42).select(range(self.max_examples))

        # Split 80% train, 20% validation
        train_size = int(len(combined) * 0.8)
        self.train = combined.select(range(train_size))
        self.validation = combined.select(range(train_size, len(combined)))

        return self

    def sample_batch(
        self, batch_size: int, split: str = "train"
    ) -> List[Dict[str, Any]]:
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
