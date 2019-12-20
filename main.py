from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D,GlobalMaxPooling1D
from DataConverter import DataConverter
import numpy as np

converter = DataConverter()
(training_inputs, training_outputs), (testing_inputs, testing_outputs) = converter.getData()

word_vector_width = 300
number_of_outputs = len(converter.intentSet)
examples = training_inputs.shape[1]



model = Sequential()
model.add(Conv1D(50, 3, input_shape=(examples, word_vector_width), activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(word_vector_width, activation="sigmoid"))
model.add(Dropout(0.2))
model.add(Dense(number_of_outputs, activation="sigmoid"))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(training_inputs, training_outputs,
          epochs=500,
          validation_data=(testing_inputs, testing_outputs))

while True:
    phrase = input("phrase : ")
    vector_phrase = np.zeros((1,50, word_vector_width))
    vector_phrase[0] = converter.get_input_array(phrase)


    predictionClass = model.predict_classes(x=vector_phrase)
    index = predictionClass[0]
    intent = converter.getIntent(index)
    print(intent)




