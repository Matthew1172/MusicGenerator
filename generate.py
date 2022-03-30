import torch
import data
import os
import time
from fractions import Fraction
from tqdm import tqdm
from music21 import *
dataset = "./dataset/set3"

seed = 1
temp = 1.0
gen_length = 500
log_interval = 200
numberOfSongs = 10

# Checkpoint location:
cwd = os.getcwd()
CHECKPOINT_DIR = 'training_checkpoints_pytorch'
CHECKPOINT_DIR = os.path.join(cwd, CHECKPOINT_DIR)
CHECKPOINT_PREFIX = 'my_ckpt.pth'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)

OUTPUTS_DIRECTORY = os.path.join(cwd, "outputs")
try:
    os.mkdir(OUTPUTS_DIRECTORY)
except FileExistsError:
    print("The {} directory already exists...".format(OUTPUTS_DIRECTORY))

OUTPUT = os.path.join(OUTPUTS_DIRECTORY, "output@"+time.asctime().replace(' ', '').replace(':', ''))
try:
    os.mkdir(OUTPUT)
except FileExistsError:
    print("The directory {} already exists...".format(OUTPUT))

GENERATION_PREFIX = "generated"
GENERATION_PREFIX = os.path.join(OUTPUT, GENERATION_PREFIX)

# Set the random seed manually for reproducibility.
torch.manual_seed(seed)
if(torch.cuda.is_available()):
    print("GPU: ",torch.cuda.get_device_name(1), " is available, Switching now.")
else:
    print("GPU is not available, using CPU.")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device is now: ", device)

#-999 : temperature has to be greater or equal 1e-3.
if temp < 1e-3:
    exit(-999)

with open(CHECKPOINT_PREFIX, 'rb') as f:
    model = torch.load(f, map_location=device)
model.eval()

dic = data.Dictionary()
try:
    dic.load_dictionary(dataset)
    dic.load_list(dataset)
except:
    print("No dictionary file available for loading. Please run the Extraction.py script before generation or training.")
    exit(-998)

ntokens = len(dic)

input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

for sn in range(1, numberOfSongs+1):
    print("Generating song {}/{}".format(sn, numberOfSongs))
    generatedSong = []
    generatedSong.append(dic.idx2word[seed])
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
            elif "Rep" in j:
                direction = j[2]
                m.append(bar.Repeat(direction=direction))
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
    except (repeat.ExpanderException):
        print("Could not output MIDI file. Badly formed repeats or repeat expressions.")