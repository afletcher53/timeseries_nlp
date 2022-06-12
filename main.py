from unicodedata import bidirectional
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import (
    Dense,
    Dropout,
    Input,
    Embedding,
    LSTM,
    Bidirectional,
)
import tensorflow as tf
from keras.layers import TextVectorization
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime
from keras.layers import TextVectorization
from os.path import exists

warnings.simplefilter(action="ignore", category=FutureWarning)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

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
AUTOTUNE = tf.data.AUTOTUNE
SEED = 2022
DATA_FILEPATH = "./data/data.csv"
REARRANGED_DATA_FILEPATH = "./data/rearranged_data.csv"
MAX_SEQUENCE_LENGTH: int = 200  # max number of words in a comment to use
MAX_VOCAB_SIZE: int = (
    10000  # how many unique words to use (i.e num rows in embedding vector)
)


def clamp(minimum: int, x: int, maximum: int):
    """Clamps an integer between a min/max"""
    return max(minimum, min(x, maximum))


def lstm_model(bidirectional: bool = False, stacked=False):
    inputs = Input(name="inputs", shape=(None,))
    layer = Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        mask_zero=False,
        name="embedding",
    )(inputs)
    if bidirectional:
        if stacked is True:
            layer = Bidirectional(LSTM(32, return_sequences=True))(layer)
        layer = Bidirectional(LSTM(32))(layer)
    else:
        if stacked is True:
            layer = LSTM(32, return_sequences=True)(layer)
        layer = LSTM(32)(layer)
    layer = Dense(256, name="FC1", activation="relu")(layer)
    layer = Dropout(0.5)(layer)
    preds = Dense(1, name="out_layer", activation="sigmoid")(layer)
    model = Model(inputs=inputs, outputs=preds)

    model.summary()

    return model


class LSTMModel:
    def __init__(self, bidirectional: bool, stacked: bool):
        self.bidirectional = bidirectional
        self.stacked = stacked
        self.model = self.create_model()

    def create_model(self):
        inputs = Input(name="inputs", shape=(None,))
        layer = Embedding(
            input_dim=MAX_VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH,
            mask_zero=False,
            name="embedding",
        )(inputs)
        if self.bidirectional:
            if self.stacked is True:
                layer = Bidirectional(LSTM(32, return_sequences=True))(layer)
            layer = Bidirectional(LSTM(32))(layer)
        else:
            if self.stacked is True:
                layer = LSTM(32, return_sequences=True)(layer)
            layer = LSTM(32)(layer)
        layer = Dense(256, name="FC1", activation="relu")(layer)
        layer = Dropout(0.5)(layer)
        preds = Dense(1, name="out_layer", activation="sigmoid")(layer)
        model = Model(inputs=inputs, outputs=preds)
        return model

    def compile_model(self):
        self.model.compile(
            loss="binary_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy", tf.keras.metrics.Precision()],
        )

    def summary(self):
        self.model.summary()


class WindowGenerator:
    def __init__(self, input_width: int, output_width: int):
        self.input_width: int = input_width
        self.output_width: int = output_width
        self.total_window_size: int = input_width + output_width
        self.minimum_day_of_year: int = 0
        self.maximum_day_of_year: int = 365

    def split_window(self, data, tv):
        all_labels: int = []
        all_input_timestep_sequences: list = []
        non_null_indexes = np.argwhere(
            data.notnull().values
        ).tolist()  # Get indexes of df where values which are not null
        data = data.head(10)  # Todo: remove in production
        for i in range(len(data.index)):
            input_sequence: str = []  # Input sequence per patient
            visit_indexes = [
                item[1] for item in non_null_indexes if item[0] == i
            ]  # Indexes of all EHR entries for patient
            for visit_index in visit_indexes:
                input_timesteps_sequence: list = []
                lower_bound = clamp(
                    self.minimum_day_of_year,
                    visit_index - self.input_width,
                    self.maximum_day_of_year,
                )
                upper_bound = clamp(
                    self.minimum_day_of_year,
                    visit_index + self.output_width,
                    self.maximum_day_of_year,
                )
                row_slice = data.iloc[[i]]
                input_sequence = row_slice.iloc[:, lower_bound + 1 : visit_index + 1]
                label_sequence = row_slice.iloc[:, visit_index + 1 : upper_bound]

                if label_sequence.isnull().all().all():
                    all_labels.append(0)
                else:
                    all_labels.append(1)
                for column in input_sequence:
                    if input_sequence[column].isnull().values.any():
                        input_timesteps_sequence.append(
                            tf.zeros([1, MAX_SEQUENCE_LENGTH], tf.int32)
                        )
                    else:
                        input_timesteps_sequence.append(tv(input_sequence[column]))

                # check that the input sequence matches the input_width
                if len(input_timesteps_sequence) != self.input_width:
                    # find difference in length and pad from the left of the input sequence
                    diff = self.input_width - len(input_timesteps_sequence)
                    for _ in range(diff):
                        input_timesteps_sequence.insert(
                            0, tf.zeros([1, MAX_SEQUENCE_LENGTH], tf.int32)
                        )

                all_input_timestep_sequences.append(input_timesteps_sequence)

        # Colapse last 3 dimensions into single sequence
        all_input_timestep_sequences = np.asarray(all_input_timestep_sequences)
        all_input_timestep_sequences = all_input_timestep_sequences.reshape(
            *all_input_timestep_sequences.shape[:-2], -1
        )
        all_input_timestep_sequences = tf.cast(
            all_input_timestep_sequences, dtype="float64"
        )
        return np.asarray(all_input_timestep_sequences), np.asarray(all_labels)


def main():
    if not exists(DATA_FILEPATH):
        raise ValueError("No datafile supplied.")

    for _ in tqdm(range(0, 3), ncols=100, desc="Loading data.."):
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

    # New dataframe to hold changed data shape - want to have columns equal to every day of the year, with each row indicating a specific patient. EHR entries are located in each cell
    if not exists(DATA_FILEPATH):
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
        print(f"------Patient data restructuring is completed ------")
        ts_df.to_csv(REARRANGED_DATA_FILEPATH)

    time_series_df = pd.read_csv(REARRANGED_DATA_FILEPATH)

    # Create TextVectorization object and fit onto training data text.
    X_train_text = df.ehr
    text_vectorization = TextVectorization(
        output_mode="int",
        split="whitespace",
        max_tokens=MAX_VOCAB_SIZE,
        output_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    text_vectorization.adapt(X_train_text)

    w1 = WindowGenerator(input_width=30, output_width=30)
    dataset, labels = w1.split_window(time_series_df, tv=text_vectorization)

    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2)
    X_train = X_train.reshape(*X_train.shape[:-2], -1)
    X_test = X_test.reshape(*X_test.shape[:-2], -1)
    print(X_train.shape, X_test.shape)

    lstm_model = LSTMModel(bidirectional=False, stacked=False)
    lstm_model.compile_model()
    lstm_model.summary()

    lstm_model.model.fit(
        X_train,
        Y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=TEST_TRAIN_SPLIT,
        callbacks=[tensorboard_callback],
    )

    lstm_model.model.evaluate(X_test, Y_test)


main()
