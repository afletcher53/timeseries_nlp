import pickle
import warnings
from os.path import exists
from datetime import datetime
import numpy
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.layers import Embedding, TextVectorization, BatchNormalization, Bidirectional
from functions.LSTMModel import LSTMModel
from functions.WindowGenerator import WindowGenerator
from keras.models import Model
from keras.layers import (
    Input,
    TimeDistributed,
    LSTM,
    Dense,
)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.optimizers import SGD


from constants import (
    BATCH_SIZE,
    DATA_FILEPATH,
    EMBEDDING_DIM,
    EMBEDDING_MATRIX_SAVE_FILE,
    EMPTY_TIMESTEP_TOKEN,
    GLOVE_300D_FILEPATH,
    LR,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    NUM_EPOCHS,
    REARRANGED_DATA_FILEPATH,
    REARRANGED_MULTI_INPUT_WINDOWED_DATA_FILEPATH,
    REARRANGED_MULTI_INPUT_WINDOWED_LABEL_FILEPATH,
    REARRANGED_SINGLE_INPUT_WINDOWED_DATA_FILEPATH,
    REARRANGED_SINGLE_INPUT_WINDOWED_LABEL_FILEPATH,
    SEED,
    TEST_TRAIN_SPLIT,
    TIME_STEP,
    VALIDATION_SPLIT,
    X_TEST_MULTI_INPUT_SAVE_FILE,
    X_TRAIN_MULTI_INPUT_SAVE_FILE,
    Y_TEST_MULTI_INPUT_SAVE_FILE,
    Y_TRAIN_MULTI_INPUT_SAVE_FILE,
)

tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
warnings.simplefilter(action="ignore", category=FutureWarning)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
AUTOTUNE = tf.data.AUTOTUNE


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


def single_timestep_predictions():
    df = load_datafile()
    time_series_df = refactor_dataframe(df)
    text_vectorization = create_textvectorisation(df)
    loaded_dataset, loaded_labels = generate_windows(time_series_df)
    dataset = vectorize_data_single_timestep(text_vectorization, loaded_dataset)
    dataset = numpy.array(dataset)
    loaded_labels = numpy.array(loaded_labels)
    X_test, Y_test, X_train, Y_train = test_train_splt(loaded_labels, dataset)
    embedding_matrix = embed_vectors(text_vectorization)

    embedding_layer = Embedding(
        MAX_VOCAB_SIZE,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False,
    )

    LSTM1 = LSTMModel(embedding_layer=embedding_layer)
    LSTM1.compile_model()

    print(LSTM1.summary())

    from keras.callbacks import ReduceLROnPlateau

    ReduceLROnPlateau = ReduceLROnPlateau(
        factor=0.1, min_lr=0.01, monitor="val_loss", verbose=1
    )

    LSTM1.model.fit(
        X_train,
        Y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=0.2,
        callbacks=[ReduceLROnPlateau],
    )

    print(LSTM1.model.evaluate(X_test, Y_test))


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


def multiple_timestep_prediction(
    load_from_save: bool = True, balance_data: bool = True
):
    if not load_from_save:
        df = load_datafile()
        time_series_df = refactor_dataframe(df)
        text_vectorization = create_textvectorisation(df)
        loaded_dataset, loaded_labels = generate_windows(time_series_df, multi=True)
        dataset = vectorize_data_multi_timestep(text_vectorization, loaded_dataset)
        loaded_labels = numpy.array(loaded_labels)
        x_test, y_test, x_train, y_train = test_train_splt(loaded_labels, dataset)
        embedding_matrix = embed_vectors(text_vectorization)
        save_variables(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            embedding_matrix=embedding_matrix,
        )

    with open(X_TRAIN_MULTI_INPUT_SAVE_FILE, "rb") as f:
        x_train = pickle.load(f)
    with open(Y_TRAIN_MULTI_INPUT_SAVE_FILE, "rb") as f:
        y_train = pickle.load(f)
    with open(X_TEST_MULTI_INPUT_SAVE_FILE, "rb") as f:
        x_test = pickle.load(f)
    with open(Y_TEST_MULTI_INPUT_SAVE_FILE, "rb") as f:
        y_test = pickle.load(f)
    with open(EMBEDDING_MATRIX_SAVE_FILE, "rb") as f:
        embedding_matrix = pickle.load(f)

    print("------Saving varaibles for reuse ------")
    print(f"X_train shape: {x_train.shape}")
    print(f"Y_train shape: {y_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"Y_test shape: {y_test.shape}")
    print(f"Embedding shape: {embedding_matrix.shape}")
    print(f"Total 0 values: {(y_train == 0).sum()}")
    print(f"Total 1 values: {(y_train == 1).sum()}")

    if balance_data:
        arr = x_train.reshape(len(x_train), -1)
        sm = SMOTE(random_state=SEED)
        x_train_bal, y_train = sm.fit_resample(arr, y_train.ravel())
        print(f"After OverSampling, the shape of train_X: {x_train_bal.shape}")
        print(f"After OverSampling, the shape of train_y: {y_train.shape}")
        print(f"After OverSampling, counts of label '1': {sum(y_train == 1)}")
        print(f"After OverSampling, counts of label '0': {sum(y_train == 0)}")
        x_train = numpy.reshape(x_train_bal, (-1, x_train.shape[1], x_train.shape[2]))

    embedding_layer = Embedding(
        MAX_VOCAB_SIZE,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False,
    )

    fit_cnn_model(x_train, y_train, x_test, y_test, embedding_layer)
    # fit_lstm_recurrent_model(x_train, y_train, x_test, y_test, embedding_layer)


def fit_cnn_model(x_train, y_train, x_test, y_test, embedding_layer):
    from keras.layers import Conv1D, Input, Dense, Flatten

    document_input = Input(
        shape=(MAX_SEQUENCE_LENGTH,),
        dtype="int32",
    )
    embedding_sequences = embedding_layer(document_input)

    x = Conv1D(filters=300, kernel_size=5, padding="valid")(embedding_sequences)
    doc_model = Model(document_input, x)
    doc_model.summary()
    input_docs = Input(
        shape=(TIME_STEP, MAX_SEQUENCE_LENGTH), name="input_docs", dtype="int32"
    )

    x = TimeDistributed(doc_model, name="token_embedding_model")(input_docs)
    x = Conv1D(filters=300, kernel_size=5, padding="valid")(x)
    x = Flatten()(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(input_docs, outputs)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.999)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=VALIDATION_SPLIT,
    )

    results = model.evaluate(x_test, y_test)
    print("test loss, test acc:", results)


def fit_lstm_recurrent_model(x_train, y_train, x_test, y_test, embedding_layer):
    document_input = Input(
        shape=(MAX_SEQUENCE_LENGTH,),
        dtype="int32",
    )
    embedding_sequences = embedding_layer(document_input)

    x = LSTM(12, return_sequences=True)(embedding_sequences)
    x = LSTM(12)(x)
    doc_model = Model(document_input, x)
    input_docs = Input(
        shape=(TIME_STEP, MAX_SEQUENCE_LENGTH), name="input_docs", dtype="int32"
    )

    x = TimeDistributed(doc_model, name="token_embedding_model")(input_docs)
    x = LSTM(12)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(input_docs, outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.999)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=VALIDATION_SPLIT,
    )

    results = model.evaluate(x_test, y_test)
    print("test loss, test acc:", results)


def vectorize_data_multi_timestep(text_vectorization, loaded_dataset):
    arr = numpy.array(loaded_dataset)
    arr[pd.isnull(arr)] = EMPTY_TIMESTEP_TOKEN
    input_samples = []
    for _, item in enumerate(
        tqdm(arr, desc="Vectoring multi timestep"),
    ):
        time_seq = []
        for _, timestep in enumerate(item):
            time_seq.append(text_vectorization(timestep))
        input_samples.append(time_seq)
    test = numpy.array(input_samples)
    return test


def main():
    multiple_timestep_prediction(load_from_save=True)
    # single_timestep_predictions()


if __name__ == "__main__":
    main()
