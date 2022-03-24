import os
from io import open
import torch
import regex as re
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
        self.dictionary = Dictionary()

        self.train = self.tokenize(os.path.join(path, 'train.abc'))
        self.valid = self.tokenize(os.path.join(path, 'valid.abc'))
        self.test = self.tokenize(os.path.join(path, 'test.abc'))

    def extract_song_snippet(self, text):
        pattern = '(^|\n\n)(.*?)\n\n'
        search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
        songs = [song[1] for song in search_results]
        return songs

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        songs = []
        with open(path, 'r', encoding="utf8") as f:
            text = f.read()
            songs = text.split("\n\n")

        m21 = []
        for i in tqdm(range(len(songs))):
            try:
                m21.append(converter.parse(songs[i]))
            except(converter.ConverterException, Exception):
                print("Converter exception on song: ", songs[i])
                continue

        #clefs = [p[0][0] for p in [[[k.sign for k in j.getElementsByClass(clef.Clef)] for j in s[1].getElementsByClass(stream.Measure)] for s in m21]]
        #keys = [p[0][0] for p in [[[(i.tonic.name, i.mode) for i in j.getElementsByClass(key.KeySignature)] for j in s[1].getElementsByClass(stream.Measure)] for s in m21]]
        #times = [p[0][0] for p in [[[(k.numerator, k.denominator) for k in j.getElementsByClass(meter.TimeSignature)] for j in s[1].getElementsByClass(stream.Measure)] for s in m21]]
        #note = [[[k.fullName for k in j.getElementsByClass(note.Note).flatten()] for j in s[1].getElementsByClass(stream.Measure)] for s in m21]

        info = [[j for j in s[1].getElementsByClass(stream.Measure).flatten()] for s in m21]
        pretty_info = []
        for s in info:
            pretty_song = []
            for m in s:
                da = ""
                if isinstance(m, note.Note):
                    da = m.fullName
                elif isinstance(m, note.Rest):
                    da = m.fullName
                elif isinstance(m, bar.Repeat):
                    da += "Rep"
                    da = m.type
                    da += "&"
                    da += m.direction
                elif isinstance(m, bar.Barline):
                    da += "Bar"
                    da = m.type
                elif isinstance(m, clef.Clef):
                    da += "Clef"
                    da = m.sign
                elif isinstance(m, key.KeySignature):
                    da += "Key"
                    da += m.tonic.name
                    da += "&"
                    da += m.mode
                elif isinstance(m, meter.TimeSignature):
                    da += "Time"
                    da += str(m.numerator)
                    da += "&"
                    da += str(m.denominator)
                else:
                    continue
                pretty_song.append(da)
                self.dictionary.add_word(da)
            pretty_info.append(pretty_song)

        # Tokenize file content
        idss = []
        for s in pretty_info:
            ids = []
            for n in s:
                ids.append(self.dictionary.word2idx[n])
            idss.append(torch.tensor(ids, dtype=torch.int64))
        ids = torch.cat(idss)
        return ids