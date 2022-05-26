import os
from io import open
import torch
from music21 import *
from common import createDictionary
from Dictionary import *
from parseAbcString import *
from sys import exit

class Corpus(object):
    def __init__(self, path, from_bin=False):
        self.m21 = []
        self.BAD_PREFIX = 'bad.abc'
        self.BAD_PREFIX = os.path.join(path, self.BAD_PREFIX)
        self.bad = 0
        self.total = 0
        self.dictionary = Dictionary()

        if(from_bin):
            try:
                self.dictionary.load_dictionary(path)
                self.dictionary.load_list(path)
            except:
                print(
                    "No dictionary file available for loading. Please run the Extraction.py script before generation or training.")
                exit(-998)

            self.train = self.tokenize_from_bin(os.path.join(path, 'train_PRETTY.pkl'))
            self.valid = self.tokenize_from_bin(os.path.join(path, 'test_PRETTY.pkl'))
            self.test = self.tokenize_from_bin(os.path.join(path, 'valid_PRETTY.pkl'))
        else:
            self.train = self.tokenize(os.path.join(path, 'train.abc'))
            self.valid = self.tokenize(os.path.join(path, 'valid.abc'))
            self.test = self.tokenize(os.path.join(path, 'test.abc'))

            self.dictionary.save_dictionary(path)
            self.dictionary.save_list(path)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        songs = []
        with open(path, 'r', encoding="ISO-8859-1") as f:
            text = f.read()
            songs = text.split("\n\n")
        self.total = len(songs)
        outputs = common.runParallel(songs, parseAbcString)
        '''Create dictionary'''
        self.dictionary = createDictionary(outputs)
        return self.tokenizeFileContent(outputs)

    def tokenize_from_bin(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            pretty_info = pickle.load(f)
        return self.tokenizeFileContent(pretty_info)

    def tokenizeFileContent(self, mySongFormat):
        # Tokenize file content
        idss = []
        for s in mySongFormat:
            ids = []
            for n in s:
                ids.append(self.dictionary.word2idx[n])
            idss.append(torch.tensor(ids, dtype=torch.int64))
        ids = torch.cat(idss)
        return ids
