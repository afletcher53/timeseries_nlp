EMBEDDING_DIM: int = 300  # Dimensions of the word vectors
MAX_VOCAB_SIZE: int = (
    10000  # how many unique words to use (i.e num rows in embedding vector)
)
MAX_SEQUENCE_LENGTH: int = 200  # max number of words in a comment to use
OOV_TOKEN: str = "<00V>"
BATCH_SIZE: int = 256
NUM_EPOCHS: int = 10
TEST_TRAIN_SPLIT: float = 0.2
VALIDATION_SPLIT: float = 0.2

SEED = 2022
DATA_FILEPATH = "./data/data.csv"
REARRANGED_DATA_FILEPATH = "./data/rearranged_data.csv"
REARRANGED_SINGLE_INPUT_WINDOWED_DATA_FILEPATH = "./data/windowed_single_input.pkl"
REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH = "./data/windowed_single_labels.pkl"
MAX_SEQUENCE_LENGTH: int = 200  # max number of words in a comment to use
MAX_VOCAB_SIZE: int = (
    10000  # how many unique words to use (i.e num rows in embedding vector)
)
LR = 1e-3

GLOVE_300D_FILEPATH = "./data/glove.6B.300d.txt"
