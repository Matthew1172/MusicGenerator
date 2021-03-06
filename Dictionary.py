import os
import pickle

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.DIC_PREFIX = 'word2idx.pkl'
        self.LIST_PREFIX = 'idx2word.pkl'

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    '''Pass the PATH to the folder e.x <set2>'''
    def save_dictionary(self, PATH):
        assert os.path.exists(PATH)
        self.DIC_PREFIX = os.path.join(PATH, self.DIC_PREFIX)
        output = open(self.DIC_PREFIX, 'wb')
        pickle.dump(self.word2idx, output)
        output.close()

    def load_dictionary(self, PATH):
        self.DIC_PREFIX = os.path.join(PATH, self.DIC_PREFIX)
        assert os.path.exists(self.DIC_PREFIX)
        pkl_file = open(self.DIC_PREFIX, 'rb')
        self.word2idx = pickle.load(pkl_file)
        pkl_file.close()

    def save_list(self, PATH):
        assert os.path.exists(PATH)
        self.LIST_PREFIX = os.path.join(PATH, self.LIST_PREFIX)
        with open(self.LIST_PREFIX, 'wb') as f:
            pickle.dump(self.idx2word, f)

    def load_list(self, PATH):
        self.LIST_PREFIX = os.path.join(PATH, self.LIST_PREFIX)
        assert os.path.exists(self.LIST_PREFIX)
        with open(self.LIST_PREFIX, 'rb') as f:
            self.idx2word = pickle.load(f)