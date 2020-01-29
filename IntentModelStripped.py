from keras.models import load_model
import numpy as np
from DataConverter import DataConverter


class IntentModelStripped:
    def __init__(self, filename):
        if filename is None:
            print("could not find the model file")
        else:
            self.model = load_model(filename)
        self.word_vector_width = 300
        self.converter = DataConverter()
        self.converter.getData()

    def get_intent(self, phrase):
        vector_phrase = np.zeros((1, self.converter.wordCap, self.word_vector_width))
        vector_phrase[0] = self.converter.get_input_array(phrase)
        prediction_vector = self.model.predict([vector_phrase])
        intent = self.converter.getIntent(prediction_vector[0])
        return intent


