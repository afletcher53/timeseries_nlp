from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, LSTM, Bidirectional
import tensorflow as tf
from constants import EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_VOCAB_SIZE
from keras.layers import (
    Dense,
    Dropout,
    Input,
    Embedding,
    LSTM,
    Bidirectional,
    SpatialDropout1D,
    Conv1D,
)


class LSTMModel:
    
    """
    Wrapper function for LSTM model
    """    
    def __init__(self, embedding_layer):
        self.embedding_layer = embedding_layer
        self.model = self.create_model(self.embedding_layer)

    def create_model(self, embedding_layer):
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
        embedding_sequences = embedding_layer(sequence_input)

        x = SpatialDropout1D(0.2)(embedding_sequences)
        x = Conv1D(64, 5, activation="relu")(x)
        x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(sequence_input, outputs)
        return model

    def compile_model(self):

        self.model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy", tf.keras.metrics.Precision()],
        )

    def summary(self):
        self.model.summary()
