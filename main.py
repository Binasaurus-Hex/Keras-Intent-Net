from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from DataConverter import DataConverter
import numpy as np
import msvcrt
import os

converter = DataConverter()
(training_inputs, training_outputs), (testing_inputs, testing_outputs) = converter.getData()

word_vector_width = 300
number_of_outputs = len(converter.intentSet)
examples = training_inputs.shape[1]

model = Sequential()
model.add(Conv1D(50, 3, input_shape=(examples, word_vector_width), activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(word_vector_width))
model.add(Dropout(0.2))
model.add(Activation("relu"))
model.add(Dense(number_of_outputs))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(training_inputs, training_outputs,
          epochs=500,
          validation_data=(testing_inputs, testing_outputs))

os.system('cls')
phrase = ""
while True:
    if msvcrt.kbhit():
        os.system('cls')
        x = msvcrt.getch()
        if x == b'\x08' and len(phrase) > 0:
            phrase = phrase[:-1]
        elif x == b'\x08':
            continue
        else:
            phrase += str(x.decode("ASCII"))
        vector_phrase = np.zeros((1, converter.wordCap, word_vector_width))
        vector_phrase[0] = converter.get_input_array(phrase)

        predictionClass = model.predict_classes(x=vector_phrase)
        index = predictionClass[0]
        intent = converter.getIntent(index)
        print("{0:50} intent:{1}".format(phrase, intent), end=" ")




