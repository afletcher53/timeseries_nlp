from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, LSTM, Bidirectional
import tensorflow as tf
from constants import EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, MAX_VOCAB_SIZE


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
            optimizer="adam",
            metrics=["accuracy", tf.keras.metrics.Precision()],
        )

    def summary(self):
        self.model.summary()
