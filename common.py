from music21 import *
import os
from Dictionary import *

DATASETS = "datasets"
DATASET = "V2-0"
bin = True

CWD = os.getcwd()
DATASETS = os.path.join(CWD, DATASETS)
DATASET = os.path.join(DATASETS, DATASET)

# Checkpoint location:
CHECKPOINT_DIR = 'training_checkpoints_pytorch'
CHECKPOINT_DIR = os.path.join(DATASET, CHECKPOINT_DIR)
CHECKPOINT_PREFIX = 'my_ckpt.pth'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)

try:
    os.mkdir(DATASETS)
except:
    print("The datasets directory {} already exists.".format(DATASETS))

try:
    os.mkdir(DATASET)
except:
    print("The new dataset directory {} already exists.".format(DATASET))

try:
    os.mkdir(CHECKPOINT_DIR)
except FileExistsError:
    print("The directory checkpoint {} already exists.".format(CHECKPOINT_DIR))

def parseAbcString(abc_song):
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

def logProcess(position, length, output):
    print("%s/%s" % (position, length))

def createDictionary(mySongFormatCombined):
    dic = Dictionary()
    for ps in mySongFormatCombined:
        for ele in ps:
            dic.add_word(ele)
    return dic