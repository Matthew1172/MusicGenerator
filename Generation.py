import os
import time
import torch
import data
from random import randint
from music21 import *
from tqdm import tqdm
from fractions import Fraction
from exceptions import *

class Generation:
    def __init__(self, **kwargs):
        self.args = {**kwargs}

        self.DATASET = self.args['dataset']

        try:
            self.rClef = eval(self.args['random_clef'])
        except KeyError:
            self.rClef = False
        except:
            self.rClef = False

        try:
            self.rKey = eval(self.args['random_key'])
        except KeyError:
            self.rKey = False
        except:
            self.rKey = False

        try:
            self.rTime = eval(self.args['random_time'])
        except KeyError:
            self.rTime = False
        except:
            self.rTime = False

        try:
            self.rSeq = eval(self.args['random_seq'])
        except KeyError:
            self.rSeq = False
        except:
            self.rSeq = False

        try:
            self.rSeqLen = int(self.args['random_seq_length'])
        except KeyError:
            self.rSeqLen = 1
        except:
            self.rSeqLen = 1

        try:
            self.temp = float(self.args['temperature'])
        except KeyError:
            self.temp = 0.85
        except:
            self.temp = 0.85

        try:
            self.gen_length = int(self.args['length'])
        except KeyError:
            self.gen_length = 100
        except:
            self.gen_length = 100

        self.log_interval = 200

        try:
            self.numberOfSongs = int(self.args['songs'])
        except KeyError:
            self.numberOfSongs = 1
        except:
            self.numberOfSongs = 1

        if (torch.cuda.is_available()):
            print("GPU: ", torch.cuda.get_device_name(1), " is available, Switching now.")
        else:
            print("GPU is not available, using CPU.")

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("Device is now: ", self.device)

        self.dic = data.Dictionary()
        try:
            self.iClef = self.args['input_clef']
        except KeyError:
            self.iClef = "Clef G"

        try:
            self.iKey = self.args['input_key']
        except KeyError:
            self.iKey = "Key 2"

        try:
            self.iTime = self.args['input_time']
        except KeyError:
            self.iTime = "Time 4 4"

        try:
            self.iSeq = self.args['input_seq'].split('$')
        except KeyError:
            self.iSeq = ["Note C 1.0"]

        self.export = []

        CWD = os.getcwd()
        self.OUTPUTS_DIRECTORY = os.path.join(CWD, "outputs")
        self.OUTPUT = os.path.join(self.OUTPUTS_DIRECTORY, "output@" + time.asctime().replace(' ', '').replace(':', ''))

        GENERATION_PREFIX = "generated"
        self.GENERATION_PREFIX = os.path.join(self.OUTPUT, GENERATION_PREFIX)

        # Checkpoint location:
        CHECKPOINT_DIR = 'training_checkpoints_pytorch'
        CHECKPOINT_DIR = os.path.join(self.DATASET, CHECKPOINT_DIR)
        CHECKPOINT_PREFIX = 'my_ckpt.pth'
        self.CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)

        '''PERFORM CHECKING'''
        assert self.temp >= 1e-3
        assert self.numberOfSongs >= 1 and self.numberOfSongs < 100
        assert self.gen_length >= 1 and self.gen_length < 1000

    def checkDataset(self):
        try:
            assert os.path.exists(self.DATASET)
        except:
            raise DatasetNotFound(self.DATASET)

    def loadModel(self):
        with open(self.CHECKPOINT_PREFIX, 'rb') as f:
            self.model = torch.load(f, map_location=self.device)
        self.model.eval()

    def loadDictionary(self):
        try:
            self.dic.load_dictionary(self.DATASET)
            self.dic.load_list(self.DATASET)
        except:
            print(
                "No dictionary file available for loading. Please run the Extraction.py script before generation or training.")
            exit(-998)

    def setInitClef(self):
        if self.rClef:
            clefs = [clef for clef in self.dic.idx2word if "Clef" in clef]
            self.iClef = clefs[randint(0, len(clefs) - 1)]

    def setInitKey(self):
        if self.rKey:
            keys = [key for key in self.dic.idx2word if "Key" in key]
            self.iKey = keys[randint(0, len(keys) - 1)]

    def setInitTime(self):
        if self.rTime:
            times = [time for time in self.dic.idx2word if "Time" in time]
            self.iTime = times[randint(0, len(times) - 1)]

    def setInitSeq(self):
        if self.rSeq:
            notes = [note for note in self.dic.idx2word if "Note" in note]
            self.iSeq = [notes[randint(0, len(notes) - 1)] for i in range(self.rSeqLen)]

    '''TODO: raise proper custom exceptions for flask server'''

    def checkInitClef(self):
        try:
            self.dic.word2idx[self.iClef]
        except KeyError:
            raise ClefNotFoundInDictionary(self.iClef)
        except:
            exit(1)

    def checkInitKey(self):
        try:
            self.dic.word2idx[self.iKey]
        except KeyError:
            raise KeyNotFoundInDictionary(self.iKey)
        except:
            exit(1)

    def checkInitTime(self):
        try:
            self.dic.word2idx[self.iTime]
        except KeyError:
            raise TimeNotFoundInDictionary(self.iTime)
        except:
            exit(1)

    def checkInitSeq(self):
        flag = False
        b = {}
        for n in self.iSeq:
            try:
                self.dic.word2idx[n]
            except KeyError:
                flag = True
                b[n] = 1
                continue
            except:
                exit(1)
        if flag:
            raise NoteNotFoundInDictionary(b)

    def generate(self):
        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.dic.word2idx[self.iTime])

        ntokens = len(self.dic)

        for sn in range(1, self.numberOfSongs + 1):
            print("Generating song {}/{}".format(sn, self.numberOfSongs))
            generatedSong = []
            if self.iClef != '':
                generatedSong.append(self.dic.idx2word[self.dic.word2idx[self.iClef]])
            if self.iKey != '':
                generatedSong.append(self.dic.idx2word[self.dic.word2idx[self.iKey]])
            if self.iTime != '':
                generatedSong.append(self.dic.idx2word[self.dic.word2idx[self.iTime]])
            if len(self.iSeq) > 0:
                for n in self.iSeq: generatedSong.append(self.dic.idx2word[self.dic.word2idx[n]])
            input = torch.Tensor([[self.dic.word2idx[word]] for word in generatedSong]).long().to(self.device)
            input = torch.cat([input, torch.randint(ntokens, (1, 1), dtype=torch.long)], 0)
            with torch.no_grad():  # no tracking history
                for i in tqdm(range(self.gen_length)):
                    output = self.model(input, False)
                    word_weights = output[-1].squeeze().div(self.temp).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    word_tensor = torch.Tensor([[word_idx]]).long().to(self.device)
                    input = torch.cat([input, word_tensor], 0)
                    word = self.dic.idx2word[word_idx]
                    generatedSong.append(word)

            p = stream.Part()
            m = stream.Measure()
            for i in generatedSong:
                if i == "|":
                    p.append(m)
                    m = stream.Measure()
                else:
                    j = i.split(" ")
                    if "Note" in j:
                        name = j[1]
                        try:
                            length = float(j[2])
                        except(ValueError):
                            length = Fraction(j[2])
                        m.append(note.Note(nameWithOctave=name, quarterLength=length))
                    elif "Rest" in j:
                        try:
                            length = float(j[2])
                        except(ValueError):
                            length = Fraction(j[2])
                        m.append(note.Rest(quarterLength=length))
                    elif "Bar" in j:
                        type = j[1]
                        m.append(bar.Barline(type=type))
                    elif "Clef" in j:
                        if j[1] == 'G':
                            m.append(clef.TrebleClef())
                        else:
                            continue
                    elif "Key" in j:
                        sharps = int(j[1])
                        m.append(key.KeySignature(sharps=sharps))
                    elif "Time" in j:
                        numerator = j[1]
                        denominator = j[2]
                        tsig = numerator + "/" + denominator
                        m.append(meter.TimeSignature(tsig))
                    else:
                        continue
            out = self.GENERATION_PREFIX + "_" + str(sn)
            self.export.append((out, p))

    def save(self):
        if len(self.export) > 0:
            try:
                os.mkdir(self.OUTPUTS_DIRECTORY)
            except FileExistsError:
                print("The {} directory already exists...".format(self.OUTPUTS_DIRECTORY))

            try:
                os.mkdir(self.OUTPUT)
            except FileExistsError:
                print("The directory {} already exists...".format(self.OUTPUT))

            for e in self.export:

                try:
                    e[1].write("text", e[0] + ".txt")
                except:
                    pass

                try:
                    e[1].write("musicxml", e[0] + ".mxl")
                except:
                    pass

                try:
                    e[1].write("midi", e[0] + ".mid")
                except repeat.ExpanderException:
                    print("Could not output MIDI file. Badly formed repeats or repeat expressions.")
                except:
                    pass
        else:
            print("No songs were generated.")