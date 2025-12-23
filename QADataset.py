from datasets import load_from_disk
from typing import List, Dict, Any
import random
from config import DATA_DIR, get_active_config


class QADataset:
    """handler for HotpotQA dataset."""

    def __init__(self, data_dir: str = None, max_examples: int = None):
        # Read from active tier config at runtime (after tier has been applied)
        self.data_dir = data_dir if data_dir is not None else DATA_DIR
        self.max_examples = (
            max_examples if max_examples is not None else get_active_config().max_examples
        )
        self.dataset = None
        self.train = None
        self.validation = None

    def load(self):
        """load dataset from disk and sample according to max_examples (80% from train, 20% from validation)."""

        self.dataset = load_from_disk(self.data_dir)
        original_train = self.dataset["train"]
        original_validation = self.dataset["validation"]

        # Calculate target sizes: 80% train, 20% validation
        train_size = int(self.max_examples * 0.8)
        val_size = self.max_examples - train_size  # Remaining 20%

        # Sample from original train split (shuffle for variety)
        if len(original_train) > train_size:
            self.train = original_train.shuffle(seed=42).select(range(train_size))
        else:
            self.train = original_train

        # Sample from original validation split (shuffle for variety)
        if len(original_validation) > val_size:
            self.validation = original_validation.shuffle(seed=42).select(range(val_size))
        else:
            self.validation = original_validation

        return self

    def sample_batch(
        self, batch_size: int, split: str = "train", seed: int = None
    ) -> List[Dict[str, Any]]:
        """
        Sample batch from dataset.

        Args:
            batch_size: Number of examples to sample
            split: "train" or "validation"
            seed: If provided, use deterministic sampling. If None, sample randomly.
                  For validation, pass a fixed seed to ensure consistent evaluation.

        Returns:
            List of examples
        """

        data = self.train if split == "train" else self.validation
        if data is None or len(data) == 0:
            return []
        sample_size = min(batch_size, len(data))

        if seed is not None:
            # Deterministic sampling for consistent evaluation
            rng = random.Random(seed)
            indices = rng.sample(range(len(data)), sample_size)
        else:
            # Random sampling (default behavior for training)
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
