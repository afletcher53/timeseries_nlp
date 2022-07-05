import pickle
import warnings
from os.path import exists
from datetime import datetime
import numpy
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.layers import Embedding
import TextVectorization from tensorflow.keras.layers.experimental.preprocessing 
from functions.WindowGenerator import WindowGenerator
from functions.dataframe import (
    create_textvectorisation,
    embed_vectors,
    generate_windows,
    load_datafile,
    refactor_dataframe,
    save_variables,
    test_train_splt,
)
from functions.models.CNN import CNN_mdl
from imblearn.over_sampling import SMOTE

from constants import (
    EMBEDDING_DIM,
    EMBEDDING_MATRIX_SAVE_FILE,
    EMPTY_TIMESTEP_TOKEN,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    SEED,
    X_TEST_MULTI_INPUT_SAVE_FILE,
    X_TRAIN_MULTI_INPUT_SAVE_FILE,
    Y_TEST_MULTI_INPUT_SAVE_FILE,
    Y_TRAIN_MULTI_INPUT_SAVE_FILE,
    LSTMSubModels,
)
from functions.models.LSTM import LSTM_mdl

tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
warnings.simplefilter(action="ignore", category=FutureWarning)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
AUTOTUNE = tf.data.AUTOTUNE


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

    lstm_mdl = LSTM_mdl(
        submodel=LSTMSubModels.LSTM,
        embedding_layer=embedding_layer,
        learning_rate=1e-13,
    )
    lstm_mdl.summary()
    lstm_mdl.fit(x_train, y_train, epochs=5)
    lstm_mdl.evaluate(x_test=x_test, y_test=y_test)

    # cnn_model = CNN_mdl(embedding_layer=embedding_layer)
    # cnn_model.fit(x_train, y_train, epochs=5)
    # cnn_model.evaluate(x_test=x_test, y_test=y_test)
    # stacked_lstm_mdl = LSTM_mdl(
    #     submodel=LSTMSubModels.STACKED_LSTM, embedding_layer=embedding_layer
    # )
    # stacked_lstm_mdl.fit(x_train, y_train, epochs=5)
    # stacked_lstm_mdl.evaluate(x_test=x_test, y_test=y_test)


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


if __name__ == "__main__":
    main()
