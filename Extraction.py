import sys
from data import *

DATASET_PATH_NAME = "NewSet"
bin = True

ex = Extractor(DATASET_PATH_NAME, bin=bin, SRC_CORPUS=sys.argv[1])
ex.extract()
