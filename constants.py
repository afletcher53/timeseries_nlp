EMBEDDING_DIM: int = 100  # Dimensions of the word vectors
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
MAX_SEQUENCE_LENGTH: int = 200  # max number of words in a comment to use
MAX_VOCAB_SIZE: int = (
    10000  # how many unique words to use (i.e num rows in embedding vector)
)
