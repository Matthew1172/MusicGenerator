import os
from io import open
import torch
from tqdm import tqdm
from music21 import *

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.bad = 0
        self.total = 0
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.abc'))
        self.valid = self.tokenize(os.path.join(path, 'valid.abc'))
        self.test = self.tokenize(os.path.join(path, 'test.abc'))

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
            try:
                m21.append(converter.parse(songs[i]))
            except(converter.ConverterException, Exception):
                #print("Converter exception on song: ", songs[i])
                self.bad+=1
                continue

        #clefs = [p[0][0] for p in [[[k.sign for k in j.getElementsByClass(clef.Clef)] for j in s[1].getElementsByClass(stream.Measure)] for s in m21]]
        #keys = [p[0][0] for p in [[[(i.tonic.name, i.mode) for i in j.getElementsByClass(key.KeySignature)] for j in s[1].getElementsByClass(stream.Measure)] for s in m21]]
        #times = [p[0][0] for p in [[[(k.numerator, k.denominator) for k in j.getElementsByClass(meter.TimeSignature)] for j in s[1].getElementsByClass(stream.Measure)] for s in m21]]
        #note = [[[k.fullName for k in j.getElementsByClass(note.Note).flatten()] for j in s[1].getElementsByClass(stream.Measure)] for s in m21]

        #info = [[j for j in s[1].getElementsByClass(stream.Measure).flatten()] for s in m21]
        #info = [[j for j in s[1]] for s in m21]
        info = [s[1].elements for s in m21]

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
                        elif isinstance(n, bar.Repeat):
                            da += "Rep"
                            da += " "
                            da += n.type
                            da += " "
                            da += n.direction
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