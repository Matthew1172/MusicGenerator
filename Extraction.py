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


result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.abc']
songs_raw = []
for f in result:
    with open(f, "r", encoding="utf8") as file:
        songs_raw.append(extract_song_snippet(file.read()))

songs = list(set([item for sub in songs_raw for item in sub if is_song(item)]))

print("Found {} songs in folder".format(len(songs)))

if SHUFFLE: random.shuffle(songs)
