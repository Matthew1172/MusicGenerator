import os
import time

import music21.musicxml.xmlObjects
import torch
import data
from random import randint
from music21 import *
from tqdm import tqdm
from fractions import Fraction
from exceptions import *
import re
from parseAbcString import *

class Generation:
    def __init__(self, **kwargs):
        self.args = {**kwargs}
        self.rSeqLen = 1
        self.numberOfSongs = 1

        self.DATASET = self.args['dataset']

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

        if (torch.cuda.is_available()):
            print("GPU: ", torch.cuda.get_device_name(1), " is available, Switching now.")
        else:
            print("GPU is not available, using CPU.")

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("Device is now: ", self.device)

        self.dic = data.Dictionary()
        try:
            self.iClef = self.args['input_clef']
            if self.iClef == "random": self.setRandInitClef()
        except KeyError:
            self.setRandInitClef()

        try:
            self.iKey = self.args['input_key']
            if self.iKey == "random": self.setRandInitKey()
        except KeyError:
            self.setRandInitKey()

        try:
            self.iTime = self.args['input_time']
            if self.iTime == "random": self.setRandInitTime()
        except KeyError:
            self.setRandInitTime()

        try:
            self.iSeq = re.split('\$\s*(?![^{}]*\})', self.args['input_seq'])
        except KeyError:
            self.setRandInitSeq()

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

    def setRandInitClef(self):
        clefs = [clef for clef in self.dic.idx2word if "Clef" in clef]
        self.iClef = clefs[randint(0, len(clefs) - 1)]

    def setRandInitKey(self):
        keys = [key for key in self.dic.idx2word if "Key" in key]
        self.iKey = keys[randint(0, len(keys) - 1)]

    def setRandInitTime(self):
        times = [time for time in self.dic.idx2word if "Time" in time]
        self.iTime = times[randint(0, len(times) - 1)]

    def setRandInitSeq(self):
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

            '''
            if self.iClef != '':
                generatedSong.append(self.dic.idx2word[self.dic.word2idx[self.iClef]])
            if self.iKey != '':
                generatedSong.append(self.dic.idx2word[self.dic.word2idx[self.iKey]])
            if self.iTime != '':
                generatedSong.append(self.dic.idx2word[self.dic.word2idx[self.iTime]])
            '''

            if len(self.iSeq) > 0:
                for n in self.iSeq: generatedSong.append(self.dic.idx2word[self.dic.word2idx[n]])

            inp = torch.Tensor([[self.dic.word2idx[word]] for word in generatedSong]).long().to(self.device)
            inp = torch.cat([inp, torch.randint(ntokens, (1, 1), dtype=torch.long, device=self.device)], 0)
            with torch.no_grad():  # no tracking history
                for i in tqdm(range(self.gen_length)):
                    output = self.model(inp, False)
                    word_weights = output[-1].squeeze().div(self.temp).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    word_tensor = torch.Tensor([[word_idx]]).long().to(self.device)
                    inp = torch.cat([inp, word_tensor], 0)
                    word = self.dic.idx2word[word_idx]
                    generatedSong.append(word)

            p = self.encode(generatedSong)
            out = self.GENERATION_PREFIX + "_" + str(sn)
            self.export.append((out, p))

    def encode(self, generatedSong):
        p = stream.Part()
        m = stream.Measure()
        for i in generatedSong:
            if i == "|":
                p.append(m)
                m = stream.Measure()
            elif "Chord" in i:
                '''Handle chord'''
                decode = re.findall('\{(.*?)\}', i)
                chords = decode[0].split("$")
                chordNotes = []
                for ch in chords:
                    temp = ch.split(' ')
                    if "Note" in temp:
                        name = temp[1]
                        try:
                            length = float(temp[2])
                        except(ValueError):
                            length = Fraction(temp[2])
                        chordNotes.append(note.Note(nameWithOctave=name, quarterLength=length))
                m.append(chord.Chord(chordNotes))
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
                    # m.append(bar.Barline(type=type))
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
        return p

    def save(self):
        if len(self.export) > 0:
            try:
                os.mkdir(self.OUTPUTS_DIRECTORY)
            except FileExistsError:
                #print("The {} directory already exists.".format(self.OUTPUTS_DIRECTORY))
                pass

            try:
                os.mkdir(self.OUTPUT)
            except FileExistsError:
                print("The directory {} already exists.".format(self.OUTPUT))
                raise CouldNotSaveInference("Trying to overwrite an existing output directory.")

            for e in self.export:
                file_name = e[0] + ".txt"
                try:
                    e[1].write("text", file_name)
                    print("Saved text file here: {}".format(file_name))
                except e:
                    print(e)
                    raise CouldNotSaveTxtFile(e)

                file_name = e[0] + ".mxl"
                try:
                    e[1].write("musicxml", file_name)
                    print("Saved mxl file here: {}".format(file_name))
                except e:
                    print(e)
                    raise CouldNotSaveMxlFile(e)

                file_name = e[0] + ".mid"
                try:
                    e[1].write("midi", file_name)
                    print("Saved midi file here: {}".format(file_name))
                except repeat.ExpanderException:
                    print("Could not output MIDI file. Badly formed repeats or repeat expressions.")
                    raise CouldNotSaveMidiFile("Could not output MIDI file. Badly formed repeats or repeat expressions.")
                except e:
                    print(e)
                    raise CouldNotSaveMidiFile(e)
        else:
            print("No songs were generated.")
            raise CouldNotSaveInference("Length of export is 0")

    def isRandomProp(self, prop):
        if "?" in prop:
            return True
        else:
            return False

    '''
    Receives a (String) time signature in my format (Time 4 4)
    Returns a (String) time signature in fractional format (4/4)
    '''
    def splitMyTime(self, time):
        t = time.split(" ")
        numerator = t[1]
        denominator = t[2]
        return numerator + "/" + denominator

    '''
    Receives a (String) time signature in fractional format (4/4)
    Returns a (String) time signature in my format (Time 4 4)
    '''
    def makeMyTime(self, time):
        t = time.split("/")
        numerator = t[0]
        denominator = t[1]
        return "Time {} {}".format(numerator, denominator)

    '''
    Receives a (String) key signature in ABC notation (Am)
    Returns a (String) key signature in my notation (Key 3)
    '''
    def makeMyKey(self, abc_key):
        return "Key {}".format(str(key.Key(abc_key).sharps))

    '''
    Receives a (String) key signature in my notation (Key 3)
    Returns a (String) key signature ABC notation (Am)
    '''
    def makeAbcKey(self, my_key):
        return key.KeySignature(int(my_key.split(" ")[1])).asKey().name

    def parseAbcToken(self, t):
        if "V:" in t:
            # get the clef
            if "name=" in t:
                try:
                    '''
                    treble
                    alto
                    tenor
                    bass
                    G (same as treble)
                    C (same as alto)
                    F (same as bass)
                    '''
                    clef = t.split("name=")[1]
                except:
                    clef = "treble"
                if clef == "?":
                    self.setRandInitClef()
                    clef = self.iClef.split(" ")[1]
                return "V:1 name={}\n".format(clef)
            else:
                return "V:1 name=tenor\n"
        elif "M:" in t:
            time = t[2:]
            # check if it is a ?
            if self.isRandomProp(time):
                self.setRandInitTime()
                tsig = self.splitMyTime(self.iTime)
                return "M:{}\n".format(str(tsig))
            else:
                self.iTime = self.makeMyTime(time)
                return "M:{}\n".format(time)
        elif "K:" in t:
            abc_key = t[2:]
            if self.isRandomProp(abc_key):
                self.setRandInitKey()
                k = self.makeAbcKey(self.iKey)
                return "K:{}\n".format(k)
            else:
                self.iKey = self.makeMyKey(abc_key)
                return "K:{}\n".format(abc_key)
        else:
            if self.isRandomProp(t):
                self.setRandInitSeq()
            return "{}\n".format(t)

    def loadDataFromAbc(self, abc):
        if ("K:" not in abc) or ("M:" not in abc):
            raise Exception

        abc_new = ""
        for ele in abc.split('\n'):
            abc_new += self.parseAbcToken(ele)

        parsed = parseAbcString(abc_new)
        self.iClef = [ele for ele in parsed if "Clef " in ele][0]
        self.iSeq = parsed+self.iSeq
