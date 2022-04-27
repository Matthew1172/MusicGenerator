import time
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

        if (torch.cuda.is_available()):
            print("GPU: ", torch.cuda.get_device_name(1), " is available, Switching now.")
        else:
            print("GPU is not available, using CPU.")

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("Device is now: ", self.device)

        self.DATASET = self.args['dataset']
        self.OUTPUTS_DIRECTORY = os.path.join(os.getcwd(), "outputs")
        self.OUTPUT = os.path.join(self.OUTPUTS_DIRECTORY, "output@" + time.asctime().replace(' ', '').replace(':', ''))
        GENERATION_PREFIX = "generated"
        self.GENERATION_PREFIX = os.path.join(self.OUTPUT, GENERATION_PREFIX)
        # Checkpoint location:
        CHECKPOINT_DIR = 'training_checkpoints_pytorch'
        CHECKPOINT_DIR = os.path.join(self.DATASET, CHECKPOINT_DIR)
        CHECKPOINT_PREFIX = 'my_ckpt.pth'
        self.CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)

        self.dic = data.Dictionary()

        self.loadModel()
        self.loadDictionary()

        try:
            self.iClef = self.args['input_clef']
            self.checkInitClef()
        except:
            self.iClef = "treble"

        try:
            self.iKey = self.args['input_key']
            self.checkInitKey()
        except:
            self.iKey = "K:D"

        try:
            self.iTime = self.args['input_time']
            self.checkInitTime()
        except:
            self.iTime = "M:4/4"

        try:
            self.iLength = self.args['input_length']
            '''TODO: create checkInitLength()'''
        except:
            self.iLength = "L:1/8"

        try:
            self.iSeq = self.args['input_seq'].split('$')
            '''User input for notes is in ABC notation. parse it and then decode the notes into out notation'''
            #self.iSeq = converter.parse(self.args['input_seq'])
        except:
            self.iSeq = ["C"]


        try:
            self.rClef = eval(self.args['random_clef'])
        except KeyError:
            self.rClef = False
        except TypeError:
            self.rClef = self.args['random_clef']
        except:
            self.rClef = False
        finally:
            if self.rClef: self.setRandInitClef()

        try:
            self.rKey = eval(self.args['random_key'])
        except KeyError:
            self.rKey = False
        except TypeError:
            self.rKey = self.args['random_key']
        except:
            self.rKey = False
        finally:
            if self.rKey: self.setRandInitKey()

        try:
            self.rTime = eval(self.args['random_time'])
        except KeyError:
            self.rTime = False
        except TypeError:
            self.rTime = self.args['random_time']
        except:
            self.rTime = False
        finally:
            if self.rTime: self.setRandInitTime()

        try:
            self.rSeqLen = int(self.args['random_seq_length'])
        except KeyError:
            self.rSeqLen = 1
        except:
            self.rSeqLen = 1

        try:
            self.rSeq = eval(self.args['random_seq'])
        except KeyError:
            self.rSeq = False
        except TypeError:
            self.rSeq = self.args['random_seq']
        except:
            self.rSeq = False
        finally:
            if self.rSeq: self.setRandInitSeq()

        try:
            self.rLength = eval(self.args['random_length'])
        except KeyError:
            self.rLength = False
        except TypeError:
            self.rLength = self.args['random_length']
        except:
            self.rLength = False
        finally:
            if self.rLength: self.setRandInitLength()

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

        self.export = []

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


        print(self.DATASET)


        try:
            self.dic.load_dictionary(self.DATASET)
            self.dic.load_list(self.DATASET)
        except:
            print(
                "No dictionary file available for loading. Please run the Extraction.py script before generation or training.")
            exit(-998)

    '''TODO: Get a random clef'''
    def setRandInitClef(self):
        clefs = [clef for clef in self.dic.idx2word if "V:" in clef]
        self.iClef = clefs[randint(0, len(clefs) - 1)]
        self.iClef = "treble"

    def setRandInitKey(self):
        keys = [key for key in self.dic.idx2word if "K:" in key]
        self.iKey = keys[randint(0, len(keys) - 1)]

    def setRandInitTime(self):
        times = [time for time in self.dic.idx2word if "M:" in time]
        self.iTime = times[randint(0, len(times) - 1)]

    def setRandInitLength(self):
        lengths = [length for length in self.dic.idx2word if "L:" in length]
        self.iLength = lengths[randint(0, len(lengths) - 1)]

    '''TODO: Get a random list of abc notes'''
    def setRandInitSeq(self):
        notes = [note for note in self.dic.idx2word if "Note" in note]
        self.iSeq = [notes[randint(0, len(notes) - 1)] for i in range(self.rSeqLen)]
        self.iSeq = ['C']

    '''TODO: parse through V: headers and look for name= value and take those clefs.'''
    def checkInitClef(self):
        return True
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

    '''
    Generation function. this method inferences the model. Once it is completed, it sets self.export.
    self.export is set to be a list of tuples, where the first element of the tuple is the file name prefix
    the song is supposed to be saved to. THe second element is a string of the ABC song to saved.
    '''
    def generate(self):
        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.dic.word2idx[self.iTime])

        ntokens = len(self.dic)

        for sn in range(1, self.numberOfSongs + 1):
            print("Generating song {}/{}".format(sn, self.numberOfSongs))
            generatedSong = []

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

    '''
    input is a String ABC token.
    output is a True or False boolean. If the token has a ABC notation header key, then return True. else return False.
    '''
    def isHeader(self, abcToken):
        head = ["L:", "M:", "K:", "P:"]
        for h in head:
            if h in abcToken:
                return True
        return False

    '''
    input is a String ABC token.
    output is a True or False boolean. If the ABC token has P: or p: in it then return True, else return False.
    '''
    def isPart(self, abcToken):
        if "P:" in abcToken or "p:" in abcToken: return True
        return False

    '''
    Takes in a list of ABC tokens.
    ex.
    ['M:3/4', 'L:1/16', 'K:Dm', 'D3', 'E', 'F', 'E', 'F', 'G', 'A2', 'd2', '|', 'd', '^c', 'e', 'c'...
    returns an string in ABC notation
    ex.
    
    '''
    def encode(self, generatedSong):
        song = ""
        for token in generatedSong:
            if self.isPart(token): song += "\n"+token+"\n"
            elif self.isHeader(token): song += "\n"+token+"\n"
            else: song += token
        return song

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

                print(e[1])

                try:
                    file_name = e[0] + ".txt"
                    song = converter.parse(e[1])
                except KeyError:
                    raise CouldNotSaveInference("Could get filename or song string from export tuple.")
                except:
                    raise CouldNotSaveInference("Could not parse ABC file.")

                try:
                    song.write("text", file_name)
                    print("Saved text file here: {}".format(file_name))
                except e:
                    print(e)
                    raise CouldNotSaveTxtFile(e)

                file_name = e[0] + ".mxl"
                try:
                    song.write("musicxml", file_name)
                    print("Saved mxl file here: {}".format(file_name))
                except e:
                    print(e)
                    raise CouldNotSaveMxlFile(e)

                file_name = e[0] + ".mid"
                try:
                    song.write("midi", file_name)
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

    def parseAbcToken(self, t):
        if "V:" in t:
            # get the clef
            if "name=" in t or "clef=" in t:
                try:
                    clef = t.split("clef=")[1]
                except:
                    clef = "treble"
                if clef == "?":
                    self.setRandInitClef()
                    clef = self.iClef.split(" ")[1]
                return "V:1 clef={}\n".format(clef)
            else:
                return "V:1 clef=treble\n"
        elif "M:" in t:
            if self.isRandomProp(t): self.setRandInitTime()
            else: self.iTime = t
            return "{}\n".format(self.iTime)
        elif "K:" in t:
            if self.isRandomProp(t): self.setRandInitKey()
            else: self.iKey = t
            return "{}\n".format(self.iKey)
        elif "L:" in t:
            if self.isRandomProp(t): self.setRandInitLength()
            else: self.iLength = t
            return "{}\n".format(self.iLength)
        else:
            if self.isRandomProp(t): self.setRandInitSeq()
            return "{}\n".format(t)

    def loadDataFromAbc(self, abc):
        if ("K:" not in abc) or ("M:" not in abc):
            raise Exception

        abc_new = ""
        for ele in abc.split('\n'):
            abc_new += self.parseAbcToken(ele)

        parsed = parseAbcString(abc_new)

        self.iClef = "treble"
        self.iSeq = parsed+self.iSeq
