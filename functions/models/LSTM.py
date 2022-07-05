from constants import (
    BATCH_SIZE,
    LR,
    MAX_SEQUENCE_LENGTH,
    MODEL_SAVE_DIR_BILSTM,
    MODEL_SAVE_DIR_BILSTM_STACKED,
    MODEL_SAVE_DIR_LSTM,
    MODEL_SAVE_DIR_LSTM_STACKED,
    NUM_EPOCHS,
    TIME_STEP,
    VALIDATION_SPLIT,
    LSTMSubModels,
)
from keras.layers import (
    Input,
    TimeDistributed,
    LSTM,
    Dense,
    Input,
    Dense,
    BatchNormalization,
)
from keras.models import Model
import tensorflow as tf
from functions.confusion_matrix import confusion_matrix


class LSTM_mdl:
    def __init__(
        self,
        embedding_layer,
        submodel: LSTMSubModels = LSTMSubModels.STACKED_LSTM,
        learning_rate: float = LR,
        neuron_num: int = 12,
    ) -> None:
        match submodel:
            case LSTMSubModels.STACKED_LSTM:
                self.stacked: bool = True
            case LSTMSubModels.BIDIRECTIONAL_STACKED_LSTM:
                self.stacked: bool = True
            case _:
                self.stacked: bool = False
        self.LR: float = learning_rate
        self.neuron_num: int = neuron_num
        self.submodel: LSTMSubModels = submodel
        self.embedding_layer = embedding_layer
        self.model = self.compile_model()

    def compile_model(self):
        document_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
        embedding_sequences = self.embedding_layer(document_input)

        if self.stacked:
            x = BatchNormalization()(embedding_sequences)
            x = LSTM(self.neuron_num, return_sequences=True)(x)
            x = LSTM(self.neuron_num)(x)
        else:
            x = BatchNormalization()(embedding_sequences)
            x = LSTM(self.neuron_num)(x)
        doc_model = Model(document_input, x)
        input_docs = Input(
            shape=(TIME_STEP, MAX_SEQUENCE_LENGTH), name="input_docs", dtype="int32"
        )

        x = TimeDistributed(doc_model, name="token_embedding_model")(input_docs)
        x = LSTM(self.neuron_num)(x)
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(input_docs, outputs)
        from tensorflow.keras.optimizers import SGD

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(
            loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
        )
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

        match self.submodel:
            case LSTMSubModels.STACKED_LSTM:
                save_dir = MODEL_SAVE_DIR_LSTM_STACKED
            case LSTMSubModels.LSTM:
                save_dir = MODEL_SAVE_DIR_LSTM
            case LSTMSubModels.BIDIRECTIONAL:
                save_dir = MODEL_SAVE_DIR_BILSTM
            case LSTMSubModels.BIDIRECTIONAL_STACKED_LSTM:
                save_dir = MODEL_SAVE_DIR_BILSTM_STACKED
        if save:
            self.model.save(save_dir)
        return history

    def summary(self):
        print(self.model.summary())

    def evaluate(self, x_test, y_test):
        res = self.model.evaluate(x=x_test, y=y_test)
        confusion_matrix(model=self.model, poison=False)
