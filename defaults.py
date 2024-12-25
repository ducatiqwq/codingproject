MODEL_NAME = "gpt2-large"
TOKENIZER_NAME = "gpt2-large"
OUTPUT_DIR = './output'
LOGGING_DIR = './logs'
LR_SCHEDULER_TYPE = "cosine"

RANDOM_SEED = 42
LOGGING_STEPS = 32
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 200
DATASET_SIZE = 10000

LEARNING_RATE = 3e-5
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 1
TEST_SIZE = 0.01