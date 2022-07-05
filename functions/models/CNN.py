from constants import (
    BATCH_SIZE,
    LR,
    MAX_SEQUENCE_LENGTH,
    MODEL_SAVE_DIR_CNN,
    NUM_EPOCHS,
    TIME_STEP,
    VALIDATION_SPLIT,
)
from functions.confusion_matrix import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    TimeDistributed,
    Dense,
    Conv1D,
    Input,
    Dense,
    Flatten,
)


class CNN_mdl:
    def __init__(self, embedding_layer, learning_rate: float = LR):
        self.learning_rate = learning_rate
        self.embedding_layer = embedding_layer
        self.model = self.compile_model()

    def compile_model(self):
        document_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
        embedding_sequences = self.embedding_layer(document_input)

        x = Conv1D(filters=300, kernel_size=5, padding="valid")(embedding_sequences)
        doc_model = Model(document_input, x)

        input_docs = Input(
            shape=(TIME_STEP, MAX_SEQUENCE_LENGTH), name="input_docs", dtype="int32"
        )

        x = TimeDistributed(doc_model, name="token_embedding_model")(input_docs)
        x = Conv1D(filters=300, kernel_size=5, padding="valid")(x)
        x = Flatten()(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(input_docs, outputs)

        opt = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.999)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    def fit(
        self,
        x_train,
        y_train,
        epochs: int = NUM_EPOCHS,
        save: bool = True,
    ):

        history = self.model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            validation_split=VALIDATION_SPLIT,
        )

        save_dir: str = MODEL_SAVE_DIR_CNN
        if save:
            self.model.save(save_dir)

        return history

    def summary(self):
        print(self.model.summary())

    def evaluate(self, x_test, y_test):
        res = self.model.evaluate(x=x_test, y=y_test)
        confusion_matrix(model=self.model, poison=False)
