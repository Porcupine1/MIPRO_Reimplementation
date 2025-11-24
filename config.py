# model configuration
MODEL_NAME = "llama3.2:3b-instruct-q4_0"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.7
MAX_TOKENS = 512

# dataset configuration
DATA_DIR = "data/hotpotqa"

# optimization parameters
N_TRIALS = 20
BATCH_SIZE = 50
N_INSTRUCTION_CANDIDATES = 10
EVAL_BATCH_SIZE = 100

# surrogate optimizer (TPE) parameters
N_STARTUP_TRIALS = 5

# evaluation metric
METRIC = "exact_match"  # or "f1"

# output paths
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"
