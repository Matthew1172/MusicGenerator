import os
from io import open
import torch
from music21 import *
import pickle
import regex as re
import random as r

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

class Extractor():
    def __init__(self, DATASET_PATH_NAME, shuffle=True, bin=False, train=0.80, test=0.1, valid=0.1, DATASETS_PATH_NAME="datasets", SRC_CORPUS=""):
        self.DATASETS_PATH_NAME = DATASETS_PATH_NAME
        self.DATASET_PATH_NAME = DATASET_PATH_NAME
        self.bin = bin
        self.SRC_CORPUS = SRC_CORPUS

        self.CWD = os.getcwd()
        self.DATASETS_PATH = os.path.join(self.CWD, self.DATASETS_PATH_NAME)
        self.DATASET_PATH = os.path.join(self.DATASETS_PATH, self.DATASET_PATH_NAME)

        self.TRAIN_PREFIX = os.path.join(self.DATASET_PATH, "train.abc")
        self.TEST_PREFIX = os.path.join(self.DATASET_PATH, "test.abc")
        self.VALID_PREFIX = os.path.join(self.DATASET_PATH, "valid.abc")
        self.TRAIN_PREFIX_PRETTY = os.path.join(self.DATASET_PATH, "train_PRETTY.pkl")
        self.TEST_PREFIX_PRETTY = os.path.join(self.DATASET_PATH, "test_PRETTY.pkl")
        self.VALID_PREFIX_PRETTY = os.path.join(self.DATASET_PATH, "valid_PRETTY.pkl")

        self.BAD_PREFIX = 'bad.abc'
        self.BAD_PREFIX = os.path.join(self.DATASET_PATH, self.BAD_PREFIX)

        self.bad = 0
        self.total = 0

        self.dictionary = Dictionary()

        self.shuffle = shuffle
        self.train = train
        self.test = test
        self.valid = valid
        assert self.train + self.test + self.valid == 1.0

        self.song_paths = []
        self.raw_songs = []
        self.songs = []

    def extract(self):
        '''Create the Datasets directory and check if it exists'''
        self.createDatasetsDirectory()

        '''Create the Dataset directory and check if it exists'''
        self.createDatasetDirectory()

        self.song_paths = self.getPaths()
        self.readRawSongs()

        self.songs = list(set([item for sub in self.raw_songs for item in sub if self.is_song(item)]))
        print("Found {} songs in folder".format(len(self.songs)))

        if self.shuffle: r.shuffle(self.songs)
        self.save()

    def createDatasetsDirectory(self):
        try:
            os.mkdir(self.DATASETS_PATH)
        except:
            print("The datasets directory {} already exists.".format(self.DATASETS_PATH_NAME))

    def createDatasetDirectory(self):
        try:
            os.mkdir(self.DATASET_PATH)
        except:
            print("The new dataset directory {} already exists.".format(self.DATASET_PATH_NAME))

    def logProcess(self, position, length, output):
        print("%s/%s" % (position, length))

    def parseAbcString(self, abc_song):
        pretty_song = []
        try:
            s = converter.parse(abc_song)[1].elements
            for m in s:
                if isinstance(m, stream.Measure):
                    pretty_song.append("|")
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
                elif isinstance(m, spanner.RepeatBracket):
                    continue
                else:
                    continue
        except:
            pass
        finally:
            return pretty_song[1:]

    def extract_song_snippet(self, text):
        pattern = '(^|\n\n)(.*?)\n\n'
        search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
        songs = [song[1] for song in search_results]
        return songs

    def is_song(self, str):
        if "X:" in str:
            return True
        else:
            return False

    def getPaths(self):
        return [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.SRC_CORPUS) for f in filenames if
                  os.path.splitext(f)[1] == '.abc']

    def readRawSongs(self):
        for f in self.song_paths:
            with open(f, "r", encoding="utf8") as file:
                self.raw_songs.append(self.extract_song_snippet(file.read()))

    def save(self):
        if self.bin:
            # outputs = common.runParallel(songs, parseAbcString, updateFunction=logProcess)
            outputs = common.runParallel(self.songs, self.parseAbcString)

            '''Create dictionary'''
            self.createDictionary(outputs)

            with open(self.TRAIN_PREFIX_PRETTY, 'wb') as f:
                pickle.dump(outputs[:int(self.train * len(outputs))], f)

            with open(self.TEST_PREFIX_PRETTY, 'wb') as f:
                pickle.dump(
                    outputs[int(self.train * len(outputs)):int(self.train * len(outputs)) + int(self.test * len(outputs))],
                    f)

            with open(self.VALID_PREFIX_PRETTY, 'wb') as f:
                pickle.dump(outputs[int(self.train * len(outputs)) + int(self.test * len(outputs)):], f)

            self.dictionary.save_dictionary(self.DATASET_PATH)
            self.dictionary.save_list(self.DATASET_PATH)
        else:
            songs_good = self.songs

            with open(self.TRAIN_PREFIX, "w") as f:
                for s in songs_good[:int(self.train * len(songs_good))]:
                    f.write(s + "\n\n")

            with open(self.TEST_PREFIX, "w") as f:
                for s in songs_good[
                         int(self.train * len(songs_good)):int(self.train * len(songs_good)) + int(self.test * len(songs_good))]:
                    f.write(s + "\n\n")

            with open(self.VALID_PREFIX, "w") as f:
                for s in songs_good[int(self.train * len(songs_good)) + int(self.test * len(songs_good)):]:
                    f.write(s + "\n\n")

    def createDictionary(self, mySongFormatCombined):
        for ps in mySongFormatCombined:
            for ele in ps:
                self.dictionary.add_word(ele)

class Corpus(Extractor):
    def __init__(self, path, bin=False):
        super(Corpus, self).__init__(path, bin=bin)
        if self.bin:
            try:
                self.dictionary.load_dictionary(self.DATASET_PATH)
                self.dictionary.load_list(self.DATASET_PATH)
            except:
                print(
                    "No dictionary file available for loading. Please run the Extraction.py script before generation or training.")
                exit(-998)

            self.train = self.tokenize_from_bin(self.TRAIN_PREFIX_PRETTY)
            self.valid = self.tokenize_from_bin(self.VALID_PREFIX_PRETTY)
            self.test = self.tokenize_from_bin(self.TEST_PREFIX_PRETTY)
        else:
            self.train = self.tokenize(self.TRAIN_PREFIX)
            self.valid = self.tokenize(self.VALID_PREFIX)
            self.test = self.tokenize(self.TEST_PREFIX)

            self.dictionary.save_dictionary(self.DATASET_PATH)
            self.dictionary.save_list(self.DATASET_PATH)

    def has_part(self, song):
        try:
            song[1]
        except IndexError:
            return False
        except exceptions21.StreamException:
            return False
        except repeat.ExpanderException:
            return False
        except:
            return False
        return True

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        songs = []
        with open(path, 'r', encoding="ISO-8859-1") as f:
            text = f.read()
            songs = text.split("\n\n")
        #self.total += len(songs)
        outputs = common.runParallel(songs, self.parseAbcString)
        '''Create dictionary'''
        self.createDictionary(outputs)
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
