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
TRAIN_CORPORA: str = os.path.join(DATA_DIR, "train_corpora.pkl")

X_TRAIN_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "X_train.pkl")
Y_TRAIN_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "Y_train.pkl")
Y_VAL_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "Y_val.pkl")
X_VAL_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "X_val.pkl")
X_TEST_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "X_test.pkl")
Y_TEST_INPUT_SAVE_FILE: str = os.path.join(DATA_DIR, "Y_test.pkl")

X_TRAIN_INPUT_SAVE_FILE_PRE_VEC: str = os.path.join(DATA_DIR, "X_train_pre_vec.pkl")
X_VAL_INPUT_SAVE_FILE_PRE_VEC: str = os.path.join(DATA_DIR, "X_val_pre_vec.pkl")
X_TEST_INPUT_SAVE_FILE_PRE_VEC: str = os.path.join(DATA_DIR, "X_test_pre_vec.pkl")

TRAIN_DS_FILE_VEC_SINGLE_TIMESTEP: str = os.path.join(DATA_DIR, "train_ds_vec_single_timestep")
VAL_DS_FILE_VEC_SINGLE_TIMESTEP: str = os.path.join(DATA_DIR, "val_ds_vec_single_timestep")
TEST_DS_FILE_VEC_SINGLE_TIMESTEP: str = os.path.join(DATA_DIR, "test_ds_vec_single_timestep")

# Datasets with the RAW text, single timestep
X_TRAIN_SINGLE_TIMESTEP_RAW: str = os.path.join(DATA_DIR, "x_train_single_timestep_raw.pkl")
X_VAL_SINGLE_TIMESTEP_RAW: str = os.path.join(DATA_DIR, "x_val_single_timestep_raw.pkl")
X_TEST_SINGLE_TIMESTEP_RAW: str = os.path.join(DATA_DIR, "x_test_single_timestep_raw.pkl")


X_TRAIN_INPUT_SAVE_FILE_VEC_MULTI_TIMESTEP: str = os.path.join(DATA_DIR, "X_train_vec_multi_timestep")
X_VAL_INPUT_SAVE_FILE_VEC_MULTI_TIMESTEP: str = os.path.join(DATA_DIR, "X_val_vec_multi_timestep")
X_TEST_INPUT_SAVE_FILE_VEC_MULTI_TIMESTEP: str = os.path.join(DATA_DIR, "X_test_vec_multi_timestep")

TRAIN_CORPORA: str = os.path.join(DATA_DIR, "train_corpora.pkl")

EMBEDDING_MATRIX_SAVE_FILE: str = os.path.join(DATA_DIR, "embedding_matrix.pkl")
GLOVE_300D_FILEPATH: str = os.path.join(DATA_DIR, "glove.6B.300d.txt")
GLOVE_100D_FILEPATH: str = os.path.join(DATA_DIR, "glove.6B.100d.txt")
FINE_TUNED_GLOVE_300D_FILEPATH: str = os.path.join(
    DATA_DIR, "fine_tuned_glove.6B.300d.txt"
)
QUERY_WORDS: str = os.path.join(
    DATA_DIR, "query_words.txt"
)  # Query words for word embedding visualisation
EMBEDDING_DIM: int = 100  # Dimensions of the word vectors
LOAD_FROM_SAVE: bool = False  # Load the rearranged df from save
TIME_STEP: int = 30
VOCAB_SAVE_FILE: str = os.path.join(DATA_DIR, "vocabulary.pkl")
MAX_SEQUENCE_LENGTH: int = 200
BALANCE_DATA: bool = True
LR: float = 1e-4
BATCH_SIZE: int = 16
NUM_EPOCHS: int = 5
VALIDATION_SPLIT: float = 0.1
