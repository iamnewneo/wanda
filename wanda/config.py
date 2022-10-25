import os

ENV = os.getenv("ENV", "dev")
configs = {
    "dev": {
        "DEVICE": "cpu",
        "BATCH_SIZE": 4,
        "TEST_BATCH_SIZE": 256,
        "MAX_EPOCHS": 1,
        "LR": 0.1,
        "N_WORKER": 0,
        "N_OPT_TRIALS": 1,
        "N_JOBS": 1,
    },
    "prod": {
        "DEVICE": "cuda",
        "BATCH_SIZE": 256,
        "TEST_BATCH_SIZE": 4096,
        "MAX_EPOCHS": 50,
        "LR": 0.00005,
        "N_WORKER": 8,
        "N_OPT_TRIALS": 100,
        "N_JOBS": 16,
    },
}

SEED = 42

BASE_PATH = os.getenv("BASE_PATH", ".")

# Trainer
N_GPU = 1
FP_PRECISION = 16

# Training
DEVICE = configs[ENV]["DEVICE"]
BATCH_SIZE = configs[ENV]["BATCH_SIZE"]
MAX_EPOCHS = configs[ENV]["MAX_EPOCHS"]
LR = configs[ENV]["LR"]
TEST_BATCH_SIZE = configs[ENV]["TEST_BATCH_SIZE"]
N_WORKER = configs[ENV]["N_WORKER"]
N_OPT_TRIALS = configs[ENV]["N_OPT_TRIALS"]
N_JOBS = configs[ENV]["N_JOBS"]
