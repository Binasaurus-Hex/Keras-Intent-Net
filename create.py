from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from DataConverter import DataConverter
import numpy as np
import msvcrt
import os

from IntentModel import IntentModel

model = IntentModel()

model.save("intent_model.h5")




