import os

# HOME_DIR:str = '/home/aaron/timeseries_nlp'
cwd = os.getcwd()
HOME_DIR: str = cwd

DATA_DIR: str = os.path.join(HOME_DIR, "data")
DATA_FILEPATH: str = os.path.join(DATA_DIR, "data.csv")
REARRANGED_DATA_FILEPATH: str = os.path.join(DATA_DIR, "rearranged_data.csv")
TEST_TRAIN_SPLIT: float = 0.2
SEED: int = 2022
MAX_VOCAB_SIZE: int = 10000
EMPTY_TIMESTEP_TOKEN: str = "[UNK]"
REARRANGED_INPUT_WINDOWED_DATA_FILEPATH: str = os.path.join(
    DATA_DIR, "windowed_input.pkl"
)
REARRANGED_INPUT_WINDOWED_LABEL_FILEPATH: str = os.path.join(
    DATA_DIR, "windowed_labels.pkl"
)
X_TRAIN_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "X_train.pkl")
Y_TRAIN_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "Y_train.pkl")
Y_VAL_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "Y_val.pkl")
X_VAL_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "X_val.pkl")
X_TEST_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "X_test.pkl")
Y_TEST_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "Y_test.pkl")

X_TRAIN_INPUT_SAVE_FILE_PRE_VEC: str = os.path.join(DATA_DIR, "X_train_pre_vec.pkl")
X_VAL_INPUT_SAVE_FILE_PRE_VEC: str = os.path.join(DATA_DIR, "X_val_pre_vec.pkl")
X_TEST_INPUT_SAVE_FILE_PRE_VEC: str = os.path.join(DATA_DIR, "X_test_pre_vec.pkl")
TRAIN_CORPORA: str = os.path.join(DATA_DIR, "train_corpora.pkl")

EMBEDDING_MATRIX_SAVE_FILE: str = os.path.join(DATA_DIR, "embedding_matrix.pkl")
GLOVE_300D_FILEPATH: str = os.path.join(DATA_DIR, "glove.6B.300d.txt")
FINE_TUNED_GLOVE_300D_FILEPATH: str = os.path.join(
    DATA_DIR, "fine_tuned_glove.6B.300d.txt"
)
QUERY_WORDS: str = os.path.join(
    DATA_DIR, "query_words.txt"
)  # Query words for word embedding visualisation
EMBEDDING_DIM: int = 300  # Dimensions of the word vectors
LOAD_FROM_SAVE: bool = False  # Load the rearranged df from save
TIME_STEP: int = 30
VOCAB_SAVE_FILE: str = os.path.join(DATA_DIR, "vocabulary.pkl")
MAX_SEQUENCE_LENGTH: int = 350
BALANCE_DATA: bool = True
LR: float = 1e-4
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 5
VALIDATION_SPLIT: float = 0.1
