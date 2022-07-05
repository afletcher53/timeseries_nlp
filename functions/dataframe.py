import pickle
from constants import (
    EMBEDDING_MATRIX_SAVE_FILE,
    X_TEST_MULTI_INPUT_SAVE_FILE,
    X_TRAIN_MULTI_INPUT_SAVE_FILE,
    Y_TEST_MULTI_INPUT_SAVE_FILE,
    Y_TRAIN_MULTI_INPUT_SAVE_FILE,
)
import pickle
from os.path import exists
from datetime import datetime
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from functions.WindowGenerator import WindowGenerator


from constants import (
    DATA_FILEPATH,
    EMBEDDING_DIM,
    EMBEDDING_MATRIX_SAVE_FILE,
    GLOVE_300D_FILEPATH,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    REARRANGED_DATA_FILEPATH,
    REARRANGED_MULTI_INPUT_WINDOWED_DATA_FILEPATH,
    REARRANGED_MULTI_INPUT_WINDOWED_LABEL_FILEPATH,
    REARRANGED_SINGLE_INPUT_WINDOWED_DATA_FILEPATH,
    REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH,
    SEED,
    TEST_TRAIN_SPLIT,
    TIME_STEP,
    X_TEST_MULTI_INPUT_SAVE_FILE,
    X_TRAIN_MULTI_INPUT_SAVE_FILE,
    Y_TEST_MULTI_INPUT_SAVE_FILE,
    Y_TRAIN_MULTI_INPUT_SAVE_FILE,
)


def save_variables(x_train, y_train, x_test, y_test, embedding_matrix):
    """Saves variables

    Args:
        x_train (_type_): _description_
        y_train (_type_): _description_
        x_test (_type_): _description_
        y_test (_type_): _description_
        embedding_matrix (_type_): _description_
    """

    print("------Saving varaibles for reuse ------")
    print(f"X_train shape: {x_train.shape}")
    print(f"Y_train shape: {y_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"Y_test shape: {y_test.shape}")

    print(f"Embedding shape: {embedding_matrix.shape}")

    with open(X_TRAIN_MULTI_INPUT_SAVE_FILE, "wb") as f:
        pickle.dump(x_train, f)
    with open(Y_TRAIN_MULTI_INPUT_SAVE_FILE, "wb") as f:
        pickle.dump(y_train, f)
    with open(X_TEST_MULTI_INPUT_SAVE_FILE, "wb") as f:
        pickle.dump(x_test, f)
    with open(Y_TEST_MULTI_INPUT_SAVE_FILE, "wb") as f:
        pickle.dump(y_test, f)

    with open(EMBEDDING_MATRIX_SAVE_FILE, "wb") as f:
        pickle.dump(embedding_matrix, f)


def embed_vectors(text_vectorization):
    embeddings_index = {}

    f = open(GLOVE_300D_FILEPATH, encoding="UTF-8")
    for line in tqdm(f, ncols=100, desc="Loading Glove Embeddings."):
        values = line.split()
        word = values[0]
        coefs = numpy.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()

    print(f"Found {len(embeddings_index)} word vectors.")

    vocabulary = text_vectorization.get_vocabulary()
    word_index = dict(zip(vocabulary, range(len(vocabulary))))
    embedding_matrix = numpy.zeros((MAX_VOCAB_SIZE, EMBEDDING_DIM))

    for word, i in tqdm(word_index.items(), desc="Embedding Matrix."):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def test_train_splt(loaded_labels, dataset):
    X_train, X_test, Y_train, Y_test = train_test_split(
        dataset,
        loaded_labels,
        test_size=TEST_TRAIN_SPLIT,
        stratify=loaded_labels,
        shuffle=True,
        random_state=SEED,
    )

    print(
        X_train.shape,
        X_test.shape,
        Y_train.shape,
        Y_test.shape,
    )
    return X_test, Y_test, X_train, Y_train


def vectorize_data_single_timestep(text_vectorization, loaded_dataset):
    dataset = list(
        map(
            text_vectorization, tqdm(loaded_dataset, ncols=100, desc="Vectorizing data")
        )
    )
    return dataset


def generate_windows(time_series_df, multi: bool = False):
    if multi:
        if not exists(REARRANGED_MULTI_INPUT_WINDOWED_DATA_FILEPATH) and not exists(
            REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH
        ):
            w1 = WindowGenerator(
                input_width=TIME_STEP, output_width=TIME_STEP, save_windows=True
            )
            w1.window_multi_input_sequence(time_series_df)
        with open(REARRANGED_MULTI_INPUT_WINDOWED_DATA_FILEPATH, "rb") as f:
            loaded_dataset = pickle.load(f)

        with open(REARRANGED_MULTI_INPUT_WINDOWED_LABEL_FILEPATH, "rb") as f:
            loaded_labels = pickle.load(f)
    else:
        if not exists(REARRANGED_SINGLE_INPUT_WINDOWED_DATA_FILEPATH) and not exists(
            REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH
        ):
            w1 = WindowGenerator(
                input_width=TIME_STEP, output_width=TIME_STEP, save_windows=True
            )
            w1.window_single_input_sequence(time_series_df)
        with open(REARRANGED_SINGLE_INPUT_WINDOWED_DATA_FILEPATH, "rb") as f:
            loaded_dataset = pickle.load(f)

        with open(REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH, "rb") as f:
            loaded_labels = pickle.load(f)

    print("------ Windowed Data Loaded ------")
    return loaded_dataset, loaded_labels


def create_textvectorisation(df):
    X_train_text = df.ehr
    text_vectorization: TextVectorization = TextVectorization(
        output_mode="int",
        split="whitespace",
        max_tokens=MAX_VOCAB_SIZE,
        output_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    text_vectorization.adapt(X_train_text)
    return text_vectorization


def refactor_dataframe(df):
    # New dataframe to hold changed data shape - want to have columns equal to every day of the year, with each row indicating a specific patient. EHR entries are located in each cell
    if not exists(REARRANGED_DATA_FILEPATH):
        doy = list(range(0, 365))  # Unsuprisingly, there are 365 days in a year
        ts_df = pd.DataFrame(
            columns=doy
        )  # add 365 day of year columns to the new dataframe
        max_patient_num: int = len(
            df.index
        )  # Assumption is that this is Z set i.e. {0, ..., 365}
        for i in tqdm(range(max_patient_num), desc="Rearranging patient data"):
            rows = df.loc[df.patient_id == i]
            for index, row in rows.iterrows():
                ts_df.at[i, row.day_of_year] = row.ehr
        print("------ Patient data restructuring is completed ------")
        ts_df.to_csv(REARRANGED_DATA_FILEPATH, index=False)

    time_series_df = pd.read_csv(REARRANGED_DATA_FILEPATH)
    return time_series_df


def load_datafile():
    if not exists(DATA_FILEPATH):
        raise ValueError("No datafile supplied.")

    for _ in tqdm(range(0, 100), ncols=100, desc="Loading data.."):
        df = pd.read_csv(DATA_FILEPATH, delimiter="\t", encoding="latin-1")
    print(f"------Loading {DATA_FILEPATH} is completed ------")

    doy = []  # Calc the day of the year for each entry in file
    for index in range(len(df)):
        d1 = datetime.strptime(df.iloc[index].date, "%Y-%m-%d %H:%M:%S")
        day_of_year = d1.timetuple().tm_yday  # returns 1 for January 1st
        doy.append(day_of_year)
    df["day_of_year"] = doy

    print(f"Total EHRs: {len(df.index)}")
    print(f"Average EHR character length: {df.ehr.apply(len).mean()}")
    return df
