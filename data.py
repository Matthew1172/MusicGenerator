import os
from io import open
import torch
from tqdm import tqdm
from music21 import *
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
        self.LIST_PREFIX = os.path.join(PATH, self.LIST_PREFIX)
        with open(self.LIST_PREFIX, 'wb') as f:
            pickle.dump(self.idx2word, f)

    def load_list(self, PATH):
        self.LIST_PREFIX = os.path.join(PATH, self.LIST_PREFIX)
        assert os.path.exists(self.LIST_PREFIX)
        with open(self.LIST_PREFIX, 'rb') as f:
            self.idx2word = pickle.load(f)

class Corpus(object):
    def __init__(self, path, from_bin=False):
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

    def has_part(self, song):
        try:
            song[1]
        except IndexError:
            return False
        return True

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        songs = []
        with open(path, 'r', encoding="ISO-8859-1") as f:
            text = f.read()
            songs = text.split("\n\n")
        self.total = len(songs)

        m21 = []
        for i in tqdm(range(len(songs))):
            #print("\n\nParsing song {}/{}. Bad: {} : \n\n {}".format(i + 1, len(songs), bad, songs[i]))
            try:
                m21.append(converter.parse(songs[i]))
            except(converter.ConverterException, Exception):
                self.bad += 1
                with open(self.BAD_PREFIX, "a", encoding="ISO-8859-1") as f:
                    f.write(songs[i] + "\n\n")
                continue

        info = [s[1].expandRepeats().elements for s in m21 if self.has_part(s)]

        pretty_info = []
        for s in info:
            pretty_song = []
            for m in s:
                if isinstance(m, stream.Measure):
                    pretty_song.append("|")
                    self.dictionary.add_word("|")
                    for n in m:
                        da = ""
                        if isinstance(n, note.Note):
                            da += "Note"
                            da += " "
                            da += n.nameWithOctave
                            da += " "
                            da += str(n.quarterLength)
                        elif isinstance(n, note.Rest):
                            da += "Rest"
                            da += " "
                            da += n.name
                            da += " "
                            da += str(n.quarterLength)
                        elif isinstance(n, bar.Barline):
                            da += "Bar"
                            da += " "
                            da += n.type
                        elif isinstance(n, clef.Clef):
                            da += "Clef"
                            da += " "
                            da += n.sign
                        elif isinstance(n, key.KeySignature):
                            da += "Key"
                            da += " "
                            da += str(n.sharps)
                        elif isinstance(n, meter.TimeSignature):
                            da += "Time"
                            da += " "
                            da += str(n.numerator)
                            da += " "
                            da += str(n.denominator)
                        else:
                            continue
                        pretty_song.append(da)
                        self.dictionary.add_word(da)
                elif isinstance(m, spanner.RepeatBracket):
                    continue
                else:
                    continue
            pretty_info.append(pretty_song[1:])

        # Tokenize file content
        idss = []
        for s in pretty_info:
            ids = []
            for n in s:
                ids.append(self.dictionary.word2idx[n])
            idss.append(torch.tensor(ids, dtype=torch.int64))
        ids = torch.cat(idss)
        return ids

    def tokenize_from_bin(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        with open(path, 'rb') as f:
            pretty_info = pickle.load(f)

        # Tokenize file content
        idss = []
        for s in pretty_info:
            ids = []
            for n in s:
                ids.append(self.dictionary.word2idx[n])
            idss.append(torch.tensor(ids, dtype=torch.int64))
        ids = torch.cat(idss)
        return ids