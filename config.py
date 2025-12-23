# model configuration
MODEL_NAME = "llama3.2:3b-instruct-q4_0"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.7
MAX_TOKENS = 512
MAX_PARALLEL_WORKERS = 4

# dataset configuration
DATA_DIR = "data/hotpotqa"
MAX_EXAMPLES = 1000  # maximum total examples to use (80% train, 20% validation)

# optimization parameters
N_TRIALS = 20
BATCH_SIZE = 35  # minibatch size for evaluation
EVAL_BATCH_SIZE = 100  # full evaluation batch size
N_INSTRUCTION_CANDIDATES = 10
MINIBATCH_FULL_EVAL_STEPS = 10  # do full eval every N minibatches

# bootstrap parameters
NUM_CANDIDATES = 30  # number of candidate demo sets to create
MAX_BOOTSTRAPPED_DEMOS = 4  # max bootstrapped demos per set
MAX_LABELED_DEMOS = 2  # max labeled demos per set (sampled from training)
BOOTSTRAP_THRESHOLD = 0.6  # minimum score to keep a bootstrapped demo

# surrogate optimizer (TPE) parameters
N_STARTUP_TRIALS = 5

# evaluation metric
METRIC = "exact_match"  # or "f1"

# output paths
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"
