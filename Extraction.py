import os
import random
import sys
import regex as re

PATH = sys.argv[1]

train = .80
test = .10
valid = .10

if train+test+valid != 1.0:
    exit(-999)

cwd = os.getcwd()
DATASETS = "dataset"
DATASETS = os.path.join(cwd, DATASETS)
try:
    os.mkdir(DATASETS)
except:
    print("The datasets directory {} already exists.".format(DATASETS))

OUTPUT_DATASET_DIR = "set2"
OUTPUT_DATASET_DIR = os.path.join(DATASETS, OUTPUT_DATASET_DIR)
try:
    os.mkdir(OUTPUT_DATASET_DIR)
except:
    print("The new dataset directory {} already exists.".format(OUTPUT_DATASET_DIR))

TRAIN_PREFIX = os.path.join(OUTPUT_DATASET_DIR, "train.abc")
TEST_PREFIX = os.path.join(OUTPUT_DATASET_DIR, "test.abc")
VALID_PREFIX = os.path.join(OUTPUT_DATASET_DIR, "valid.abc")

def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    return songs

result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.abc']
songs_raw = []
for f in result:
    with open(f, "r", encoding="utf8") as file:
        songs_raw.append(extract_song_snippet(file.read()))

songs = list(set([item for sub in songs_raw for item in sub if "X:" in item]))

print("Found {} songs in folder".format(len(songs)))

random.shuffle(songs)

with open(TRAIN_PREFIX, "w") as f:
    for s in songs[:int(train*len(songs))]:
        f.write(s+"\n\n")

with open(TEST_PREFIX, "w") as f:
    for s in songs[int(train * len(songs)):int(train * len(songs))+int(test * len(songs))]:
        f.write(s+"\n\n")

with open(VALID_PREFIX, "w") as f:
    for s in songs[int(train * len(songs))+int(test * len(songs)):]:
        f.write(s+"\n\n")
