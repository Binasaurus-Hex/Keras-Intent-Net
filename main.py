import msvcrt
import os

from IntentModel import IntentModel

model = IntentModel()

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




