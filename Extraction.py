import os
import random
import sys
import regex as re
from music21 import *
import pickle
from common import DATASET, createDictionary
from parseAbcString import *

SHUFFLE = True
PATH = sys.argv[1]
train = .80
test = .10
valid = .10

if train+test+valid != 1.0:
    exit(-999)

TRAIN_PREFIX = os.path.join(DATASET, "train.abc")
TEST_PREFIX = os.path.join(DATASET, "test.abc")
VALID_PREFIX = os.path.join(DATASET, "valid.abc")
TRAIN_PREFIX_PRETTY = os.path.join(DATASET, "train_PRETTY.pkl")
TEST_PREFIX_PRETTY = os.path.join(DATASET, "test_PRETTY.pkl")
VALID_PREFIX_PRETTY = os.path.join(DATASET, "valid_PRETTY.pkl")

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

result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.abc']
songs_raw = []
for f in result:
    with open(f, "r", encoding="utf8") as file:
        #songs_raw.append(extract_song_snippet(file.read()))
        #songs_raw.append(file.read())
        songs_raw.append(["X:"+s for s in re.split("X:",file.read()) if len(s) > 1])

songs = list(set([item.rstrip() for sub in songs_raw for item in sub if is_song(item)]))

print("Found {} songs in folder".format(len(songs)))

if SHUFFLE: random.shuffle(songs)

if bin:
    #outputs = common.runParallel(songs, parseAbcString, updateFunction=logProcess, updateSendsIterable=True)
    outputs = common.runParallel(songs, parseAbcString)
    print("Done parsing.")

    '''Create dictionary'''
    dic = createDictionary(outputs)
    with open(TRAIN_PREFIX_PRETTY, 'wb') as f:
        pickle.dump(outputs[:int(train * len(outputs))], f)

    with open(TEST_PREFIX_PRETTY, 'wb') as f:
        pickle.dump(
            outputs[int(train * len(outputs)):int(train * len(outputs)) + int(test * len(outputs))],
            f)

    with open(VALID_PREFIX_PRETTY, 'wb') as f:
        pickle.dump(outputs[int(train * len(outputs)) + int(test * len(outputs)):], f)

    dic.save_dictionary(DATASET)
    dic.save_list(DATASET)
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
