import os
import random
import sys
import regex as re
from music21 import *
import data
import pickle
OUTPUT_DATASET_DIR = "set2"
save_to_bin = True

SHUFFLE = True
PATH = sys.argv[1]
train = .80
test = .10
valid = .10

if train+test+valid != 1.0:
    exit(-999)

CWD = os.getcwd()
DATASETS = "dataset"
DATASETS = os.path.join(CWD, DATASETS)
try:
    os.mkdir(DATASETS)
except:
    print("The datasets directory {} already exists.".format(DATASETS))

OUTPUT_DATASET_DIR = os.path.join(DATASETS, OUTPUT_DATASET_DIR)
try:
    os.mkdir(OUTPUT_DATASET_DIR)
except:
    print("The new dataset directory {} already exists.".format(OUTPUT_DATASET_DIR))

TRAIN_PREFIX = os.path.join(OUTPUT_DATASET_DIR, "train.abc")
TEST_PREFIX = os.path.join(OUTPUT_DATASET_DIR, "test.abc")
VALID_PREFIX = os.path.join(OUTPUT_DATASET_DIR, "valid.abc")
TRAIN_PREFIX_PRETTY = os.path.join(OUTPUT_DATASET_DIR, "train_PRETTY.pkl")
TEST_PREFIX_PRETTY = os.path.join(OUTPUT_DATASET_DIR, "test_PRETTY.pkl")
VALID_PREFIX_PRETTY = os.path.join(OUTPUT_DATASET_DIR, "valid_PRETTY.pkl")

def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    return songs

def is_song(str):
    if "X:" in str:
        return True
    else:
        return False

def has_part(song):
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

def logProcess(position, length, output):
    print("%s/%s" % (position, length))

def parseAbcString(abc_song):
    pretty_song = []
    try:
        s = converter.parse(abc_song)[1].elements
        for m in s:
            if isinstance(m, stream.Measure):
                pretty_song.append("|")
                dic.add_word("|")
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
                    dic.add_word(da)
            elif isinstance(m, spanner.RepeatBracket):
                continue
            else:
                continue
    except:
        pass
    finally:
        return pretty_song[1:]

result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.abc']
songs_raw = []
for f in result:
    with open(f, "r", encoding="utf8") as file:
        songs_raw.append(extract_song_snippet(file.read()))

songs = list(set([item for sub in songs_raw for item in sub if is_song(item)]))

print("Found {} songs in folder".format(len(songs)))

if SHUFFLE: random.shuffle(songs)

if save_to_bin:
    dic = data.Dictionary()

    #outputs = common.runParallel(songs, parseAbcString, updateFunction=logProcess)
    outputs = common.runParallel(songs, parseAbcString)

    with open(TRAIN_PREFIX_PRETTY, 'wb') as f:
        pickle.dump(outputs[:int(train * len(outputs))], f)

    with open(TEST_PREFIX_PRETTY, 'wb') as f:
        pickle.dump(
            outputs[int(train * len(outputs)):int(train * len(outputs)) + int(test * len(outputs))],
            f)

    with open(VALID_PREFIX_PRETTY, 'wb') as f:
        pickle.dump(outputs[int(train * len(outputs)) + int(test * len(outputs)):], f)

    dic.save_dictionary(OUTPUT_DATASET_DIR)
    dic.save_list(OUTPUT_DATASET_DIR)
else:
    songs_good = songs

    with open(TRAIN_PREFIX, "w") as f:
        for s in songs_good[:int(train * len(songs_good))]:
            f.write(s + "\n\n")

    with open(TEST_PREFIX, "w") as f:
        for s in songs_good[int(train * len(songs_good)):int(train * len(songs_good)) + int(test * len(songs_good))]:
            f.write(s + "\n\n")

    with open(VALID_PREFIX, "w") as f:
        for s in songs_good[int(train * len(songs_good)) + int(test * len(songs_good)):]:
            f.write(s + "\n\n")
