import os
import random
import sys
import regex as re
from music21 import *
import data
import pickle
from tqdm import tqdm
OUTPUT_DATASET_DIR = "set1"
save_to_bin = True

SHUFFLE = True
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
    m21 = []
    songs_good = []
    bad = 0
    BAD_PREFIX = "bad.abc"
    BAD_PREFIX = os.path.join(OUTPUT_DATASET_DIR, BAD_PREFIX)
    for i in range(len(songs)):
        print("\n\nParsing song {}/{}. Bad: {} : \n\n {}".format(i + 1, len(songs), bad, songs[i]))
        try:
            m21.append(converter.parse(songs[i]))
            songs_good.append(songs[i])
        except(converter.ConverterException, Exception):
            bad += 1
            with open(BAD_PREFIX, "a") as f:
                f.write(songs[i] + "\n\n")
            continue

    # info = [s[1].expandRepeats().elements for s in m21 if self.has_part(s)]
    print("Expanding repeats on songs.")
    info = []
    for i in tqdm(range(len(m21))):
        try:
            info.append(m21[i][1].expandRepeats().elements)
        except IndexError:
            continue
        except exceptions21.StreamException:
            continue
        except repeat.ExpanderException:
            continue
        except:
            continue

    pretty_info = []
    for s in info:
        pretty_song = []
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
        pretty_info.append(pretty_song[1:])

        with open(TRAIN_PREFIX_PRETTY, 'wb') as f:
            pickle.dump(pretty_info[:int(train * len(pretty_info))], f)

        with open(TEST_PREFIX_PRETTY, 'wb') as f:
            pickle.dump(
                pretty_info[int(train * len(pretty_info)):int(train * len(pretty_info)) + int(test * len(pretty_info))],
                f)

        with open(VALID_PREFIX_PRETTY, 'wb') as f:
            pickle.dump(pretty_info[int(train * len(pretty_info)) + int(test * len(pretty_info)):], f)

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
