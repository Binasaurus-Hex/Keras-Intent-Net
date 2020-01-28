from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from DataConverter import DataConverter
import numpy as np
import msvcrt
import os

from Model import Model

model = Model()

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
        print("{0:50} intent:{1}".format(phrase, model.getIntent(phrase)), end=" ")




