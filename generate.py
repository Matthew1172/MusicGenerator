import torch
import data
import os
import time
from fractions import Fraction
from tqdm import tqdm
from music21 import *
from random import randint
import argparse
DATASET = "set3"

parser = argparse.ArgumentParser(description='Music Generator by Matthew Pecko')
# Model parameters.
parser.add_argument('--temperature', type=float, default=0.85,
                    help='temperature - higher will increase diversity')
parser.add_argument('--length', type=int, default=100,
                    help='Length of song to be generated')
parser.add_argument('--songs', type=int, default=3,
                    help='Number of songs to generate')

parser.add_argument('--random-clef', type=bool, default=False,
                    help='Assign a random clef')
parser.add_argument('--random-key', type=bool, default=True,
                    help='Assign a random key signature')
parser.add_argument('--random-time', type=bool, default=True,
                    help='Assign a random time signature')
parser.add_argument('--random-seq', type=bool, default=True,
                    help='Assign a random sequence of notes')
parser.add_argument('--random-seq-length', type=int, default=1,
                    help='Number of random notes to create')

parser.add_argument('--input-clef', type=str, default="Clef G",
                    help='Assign a clef ("Clef G", "Clef F"). Only supports treble clef as of version 1.0')
parser.add_argument('--input-time', type=str, default="Time 4 4",
                    help='Assign a time signature ("Time 4 4", "Time 2 4", "Time 3 4", "Time 27 16")')
parser.add_argument('--input-key', type=str, default="Key 2",
                    help='Assign a key signature ("Key 2", "Key 1", "Key -6", "Key 8")')
parser.add_argument('--input-seq', type=str, default="Note C 1.0",
                    help='Assign a sequence of notes. Use $ as delimiter. ("Note C 1.0", "Note C 1.0$Note F#4 1/3$Note C4 0.5")')
args = parser.parse_args()

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater than 1e-3.")

if args.songs < 1 or args.songs > 100:
    parser.error("--songs has to be greater than 0 and less than 100.")

if args.length < 1 or args.length > 1000:
    parser.error("--length has to be greater than 0 and less than 1000.")

if args.random_seq and (args.random_seq_length < 1 or args.random_seq_length > 1000 or args.random_seq_length > args.length):
    parser.error("if --random-seq is true then --random-seq-length must be less than --length, and has to be greater than 0 and less than 1000.")

rClef = args.random_clef
rKey = args.random_key
rTime = args.random_time
rSeq = args.random_seq
rSeqLen = args.random_seq_length

'''
song = [<clef>, <key>, <time>, <note>, ...]
'''
''' seed = Note C 1.0'''
#seed = 492
temp = args.temperature
gen_length = args.length
log_interval = 200
numberOfSongs = args.songs

# Checkpoint location:
CWD = os.getcwd()
#Dataset location
DATASETS = "dataset"
DATASETS = os.path.join(CWD, DATASETS)
assert os.path.exists(DATASETS)
DATASET = os.path.join(DATASETS, DATASET)
assert os.path.exists(DATASET)
# Checkpoint location:
CHECKPOINT_DIR = 'training_checkpoints_pytorch'
CHECKPOINT_DIR = os.path.join(DATASET, CHECKPOINT_DIR)
CHECKPOINT_PREFIX = 'my_ckpt.pth'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)

OUTPUTS_DIRECTORY = os.path.join(CWD, "outputs")
OUTPUT = os.path.join(OUTPUTS_DIRECTORY, "output@"+time.asctime().replace(' ', '').replace(':', ''))

GENERATION_PREFIX = "generated"
GENERATION_PREFIX = os.path.join(OUTPUT, GENERATION_PREFIX)

if(torch.cuda.is_available()):
    print("GPU: ",torch.cuda.get_device_name(1), " is available, Switching now.")
else:
    print("GPU is not available, using CPU.")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device is now: ", device)

with open(CHECKPOINT_PREFIX, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

dic = data.Dictionary()
try:
    dic.load_dictionary(DATASET)
    dic.load_list(DATASET)
except:
    print("No dictionary file available for loading. Please run the Extraction.py script before generation or training.")
    exit(-998)

'''TODO: Find a random clef in dictionary and set it to iClef'''
if rClef:
    clefs = [clef for clef in dic.idx2word if "Clef" in clef]
    iClef = clefs[randint(0, len(clefs))]
else:
    iClef = args.input_clef

if rKey:
    keys = [key for key in dic.idx2word if "Key" in key]
    iKey = keys[randint(0, len(keys))]
else:
    iKey = args.input_key

if rTime:
    times = [time for time in dic.idx2word if "Time" in time]
    iTime = times[randint(0, len(times))]
else:
    iTime = args.input_time

if rSeq:
    notes = [note for note in dic.idx2word if "Note" in note]
    iSeq = [notes[randint(0, len(notes))] for i in range(rSeqLen)]
else:
    iSeq = args.input_seq.split('$')
    #iSeq = ["Note C 1.0"]
    #iSeq = []

try:
    dic.word2idx[iClef]
except KeyError:
    parser.error("The clef {} was not found in the dictionary.".format(iClef))
except:
    exit(1)

try:
    dic.word2idx[iKey]
except KeyError:
    parser.error("The key signature {} was not found in the dictionary.".format(iKey))
except:
    exit(1)

try:
    dic.word2idx[iTime]
except KeyError:
    parser.error("The time signature {} was not found in the dictionary.".format(iTime))
except:
    exit(1)

flag = False
b = []
for n in iSeq:
    try:
        dic.word2idx[n]
    except KeyError:
        flag = True
        b.append(n)
        continue
    except:
        exit(1)
if flag:
    parser.error("One or more notes in the input sequence was not found in the dictionary. {}".format(b))

# Set the random seed manually for reproducibility.
torch.manual_seed(dic.word2idx[iTime])

ntokens = len(dic)

for sn in range(1, numberOfSongs+1):
    print("Generating song {}/{}".format(sn, numberOfSongs))
    generatedSong = []
    if iClef != '':
        generatedSong.append(dic.idx2word[dic.word2idx[iClef]])
    if iKey != '':
        generatedSong.append(dic.idx2word[dic.word2idx[iKey]])
    if iTime != '':
        generatedSong.append(dic.idx2word[dic.word2idx[iTime]])
    if len(iSeq) > 0:
        for n in iSeq: generatedSong.append(dic.idx2word[dic.word2idx[n]])
    input = torch.Tensor([[dic.word2idx[word]] for word in generatedSong]).long().to(device)
    input = torch.cat([input, torch.randint(ntokens, (1, 1), dtype=torch.long)], 0)
    with torch.no_grad():  # no tracking history
        for i in tqdm(range(gen_length)):
            output = model(input, False)
            word_weights = output[-1].squeeze().div(temp).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input = torch.cat([input, word_tensor], 0)
            word = dic.idx2word[word_idx]
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
                length = float(j[2])
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

    try:
        os.mkdir(OUTPUTS_DIRECTORY)
    except FileExistsError:
        print("The {} directory already exists...".format(OUTPUTS_DIRECTORY))

    try:
        os.mkdir(OUTPUT)
    except FileExistsError:
        print("The directory {} already exists...".format(OUTPUT))

    out = GENERATION_PREFIX + "_"+str(sn)

    try:
        p.write("text", out + ".txt")
    except:
        pass

    try:
        p.write("musicxml", out + ".mxl")
    except:
        pass

    try:
        p.write("midi", out + ".mid")
    except repeat.ExpanderException:
        print("Could not output MIDI file. Badly formed repeats or repeat expressions.")
    except:
        pass