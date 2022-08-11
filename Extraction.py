import os
import sys
import regex as re
import pickle
from common import DATASET_PREFIX_PRETTY
from parseAbcString import *

PATH = sys.argv[1]

file_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.abc']
read = ""
for f in file_paths:
    with open(f, "r", encoding="utf8") as file:
        read += file.read()

songs = [i for i in re.split("\n\n", read) if "X:" in i]
print("Found {} songs in folder".format(len(songs)))
outputs = common.runParallel(songs, parseAbcString)
print("Done parsing.")

try:
    with open(DATASET_PREFIX_PRETTY, 'wb') as f:
        pickle.dump(outputs, f)
except FileNotFoundError:
    '''TODO: create the directory that doesn't exist'''
    pass
except:
    pass