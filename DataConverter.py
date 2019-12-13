import json
import numpy as np
import random
import gensim

'''
class to convert the list of phrases and associated intents into usable test data
also will convert a provided phrase to input data
'''


class DataConverter:
    def __init__(self):
        self.trainingFile = "phraseData.json"
        self.testingFile = "testPhraseData.json"
        self.wordSet = []
        self.intentSet = []
        self.model = gensim.models.KeyedVectors.load("word2Vec.model")

    '''
    @:returns {np.ndarray} inputTraining,outputTraining
    these are the input and output training matricies generated from the phraseData.json file
    '''

    def getData(self):
        training_inputs, training_outputs = self.read_data(self.trainingFile)
        test_inputs, test_outputs = self.read_data(self.testingFile)
        return (training_inputs, training_outputs), (test_inputs, test_outputs)

    def read_data(self, filename):
        json_data = self.get_json(filename)
        self.wordSet, self.intentSet = self.generate_index_lists(json_data)
        examples = self.get_examples(json_data)
        input_training, output_training = self.getDataMatrices(examples)
        return input_training, output_training

    def clean_word(self, word):
        return word.lower()

    '''
    @:returns {JSON} - returns json object from phraseData.json
    '''

    def get_json(self, filename):
        with open(filename) as phraseFile:
            return json.load(phraseFile)

    '''
    @:param {JSON} json_data - json object to turn into 2d example array
    @:returns {String[][]} - 2d array of examples, where each example is the intent, and the phrase that it relates to
    '''

    def get_examples(self, json_data):
        example_list = []
        for intent in json_data:
            for phrase in json_data[intent]:
                example = [intent, phrase]
                example_list.append(example)
        random.shuffle(example_list)
        return example_list

    '''
    @:param {JSON} json_data - json object from phraseData.json
    '''

    def generate_index_lists(self, json_data):
        words = []
        intents = []
        for intent in json_data:
            intents.append(intent)
            for phrase in json_data[intent]:
                for word in phrase.split():
                    words.append(self.clean_word(word))
        return list(dict.fromkeys(words)), list(dict.fromkeys(intents))

    def get_input_array(self, phrase):
        phraseList = phrase.split()
        inputArray = np.zeros(300)
        for word in phraseList:
            try:
                word = self.clean_word(word)
                word_vec = self.model.get_vector(word)
                inputArray += word_vec
            except ValueError:
                continue
            except KeyError:
                print(word)
                continue
        return inputArray

    def getOutputArray(self, intent):
        outputArray = np.zeros(len(self.intentSet))
        try:
            index = self.intentSet.index(intent)
            outputArray[index] = 1
        except ValueError:
            pass
        return outputArray

    def getDataMatrices(self, examples):
        input_matrix = []
        output_matrix = []
        for example in examples:
            input_matrix.append(self.get_input_array(example[1]))
            output_matrix.append(self.getOutputArray(example[0]))
        return np.array(input_matrix), np.array(output_matrix)

    def getIntent(self, index):
        return self.intentSet[index]
