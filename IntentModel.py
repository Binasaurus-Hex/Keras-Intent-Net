from tensorflow_core.python.keras.layers import concatenate
from tensorflow_core.python.keras import Input, Model

from DataConverter import DataConverter
from tensorflow_core.python.keras.layers import Dense, Dropout, Activation
from tensorflow_core.python.keras.layers import Conv1D, GlobalMaxPooling1D
import numpy as np


class IntentModel:
    def __init__(self):
        self.converter = DataConverter()
        (training_inputs, training_outputs), (testing_inputs, testing_outputs) = self.converter.getData()

        self.word_vector_width = 300
        self.number_of_outputs = len(self.converter.intentSet)
        self.examples = training_inputs.shape[1]

        self.model = self.getModel(training_inputs, training_outputs, testing_inputs, testing_outputs)
        self.model.save("intentModel.h5")

    def getModel(self, training_inputs, training_outputs, testing_inputs, testing_outputs):
        input_tensor = Input(shape=(self.examples, self.word_vector_width))
        conv_filters = []
        for kernel_len in [1, 2, 3]:
            conv = Conv1D(4, kernel_len, activation="relu")(input_tensor)
            pooling = GlobalMaxPooling1D()(conv)
            conv_filters.append(pooling)
        merged = concatenate(conv_filters)
        input_dense = Dense(self.word_vector_width)(merged)
        dropout = Dropout(0.2)(input_dense)
        input_activation = Activation("relu")(dropout)
        output_dense = Dense(self.number_of_outputs, activation="sigmoid")(input_activation)
        model = Model(inputs=[input_tensor], outputs=[output_dense])
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(training_inputs, training_outputs,
                  epochs=700,
                  validation_data=(testing_inputs, testing_outputs))

        return model

    def getIntent(self, phrase):
        vector_phrase = np.zeros((1, self.converter.wordCap, self.word_vector_width))
        vector_phrase[0] = self.converter.get_input_array(phrase)
        prediction_vector = self.model.predict([vector_phrase])
        intent = self.converter.getIntent(prediction_vector[0])
        return intent
