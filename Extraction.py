import sys
from data import *

DATASET_PATH_NAME = "set3"
from_bin = False

ex = Extractor(DATASET_PATH_NAME, SRC_CORPUS=sys.argv[1])
ex.extract()
