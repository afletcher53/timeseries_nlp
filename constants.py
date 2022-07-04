EMBEDDING_DIM: int = 300  # Dimensions of the word vectors
MAX_VOCAB_SIZE: int = (
    10000  # how many unique words to use (i.e num rows in embedding vector)
)
MAX_SEQUENCE_LENGTH: int = 200  # max number of words in a comment to use
OOV_TOKEN: str = "<00V>"
BATCH_SIZE: int = 128
NUM_EPOCHS: int = 20
TEST_TRAIN_SPLIT: float = 0.2
VALIDATION_SPLIT: float = 0.2

SEED: int = 2022
DATA_FILEPATH: str = "./data/data.csv"
REARRANGED_DATA_FILEPATH: str = "./data/rearranged_data.csv"
REARRANGED_SINGLE_INPUT_WINDOWED_DATA_FILEPATH: str = "./data/windowed_single_input.pkl"
REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH: str = (
    "./data/windowed_single_labels.pkl"
)
REARRANGED_MULTI_INPUT_WINDOWED_DATA_FILEPATH: str = "./data/windowed_MULTI_input.pkl"
REARRANGED_MULTI_INPUT_WINDOWED_LABEL_FILEPATH: str = "./data/windowed_MULTI_labels.pkl"
MAX_SEQUENCE_LENGTH: int = 200  # max number of words in a comment to use
MAX_VOCAB_SIZE: int = (
    10000  # how many unique words to use (i.e num rows in embedding vector)
)
LR: float = 1e-3

GLOVE_300D_FILEPATH: str = "./data/glove.6B.300d.txt"
EMPTY_TIMESTEP_TOKEN: str = "<EMPTY>"
TIME_STEP: int = 30

X_TRAIN_MULTI_INPUT_SAVE_FILE: str = "./data/X_train_MULTI.pkl"
Y_TRAIN_MULTI_INPUT_SAVE_FILE: str = "./data/Y_train_MULTI.pkl"
Y_VAL_MULTI_INPUT_SAVE_FILE: str = "./data/Y_val_MULTI.pkl"
X_VAL_MULTI_INPUT_SAVE_FILE: str = "./data/X_val_MULTI.pkl"
X_TEST_MULTI_INPUT_SAVE_FILE: str = "./data/X_test_MULTI.pkl"
Y_TEST_MULTI_INPUT_SAVE_FILE: str = "./data/Y_test_MULTI.pkl"
EMBEDDING_MATRIX_SAVE_FILE: str = "./data/embedding_matrix.pkl"
