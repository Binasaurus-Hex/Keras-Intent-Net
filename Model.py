from DataConverter import DataConverter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
import numpy as np


class Model:
    def __init__(self):
        self.converter = DataConverter()
        (training_inputs, training_outputs), (testing_inputs, testing_outputs) = self.converter.getData()

        self.word_vector_width = 300
        self.number_of_outputs = len(self.converter.intentSet)
        self.examples = training_inputs.shape[1]

        self.model = Sequential()
        self.model.add(Conv1D(50, 3, input_shape=(self.examples, self.word_vector_width), activation="relu"))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(self.word_vector_width))
        self.model.add(Dropout(0.2))
        self.model.add(Activation("relu"))
        self.model.add(Dense(self.number_of_outputs))
        self.model.add(Activation("sigmoid"))
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.fit(training_inputs, training_outputs,
                       epochs=500,
                       validation_data=(testing_inputs, testing_outputs))

    def getIntent(self, phrase):
        vector_phrase = np.zeros((1, self.converter.wordCap, self.word_vector_width))
        vector_phrase[0] = self.converter.get_input_array(phrase)

        predictionClass = self.model.predict_classes(x=vector_phrase)
        index = predictionClass[0]
        intent = self.converter.getIntent(index)
        return intent
