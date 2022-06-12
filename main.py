from unicodedata import bidirectional
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.layers import TextVectorization
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime
from keras.layers import TextVectorization
from os.path import exists

from constants import (
    BATCH_SIZE,
    DATA_FILEPATH,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    NUM_EPOCHS,
    REARRANGED_DATA_FILEPATH,
    TEST_TRAIN_SPLIT,
)
from functions.LSTMModel import LSTMModel
from functions.WindowGenerator import WindowGenerator

warnings.simplefilter(action="ignore", category=FutureWarning)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
AUTOTUNE = tf.data.AUTOTUNE


def main():
    """
    1 - Load datafile
    2 - Rearrange datafile if needed, if present skip
    3 - Create TextVectorization
    4 - Window Data
    5 - Fit and Train Model
    """

    """
    1. Load Datafile
    """
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

    """
    2. Create rearranged datafile, if needed
    """
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

    """
    3. Create TextVectorization object
    """
    X_train_text = df.ehr
    text_vectorization = TextVectorization(
        output_mode="int",
        split="whitespace",
        max_tokens=MAX_VOCAB_SIZE,
        output_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    text_vectorization.adapt(X_train_text)

    """
    4. Window data with WindowGenerator
    """
    w1 = WindowGenerator(input_width=30, output_width=30)
    dataset, labels = w1.split_window(time_series_df, tv=text_vectorization)

    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=0.2)
    X_train = X_train.reshape(*X_train.shape[:-2], -1)
    X_test = X_test.reshape(*X_test.shape[:-2], -1)
    print(X_train.shape, X_test.shape)

    """
    5. Fit and train models
    """
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


if __name__ == "__main__":
    main()
