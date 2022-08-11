import os

DATASETS = "datasets"
DATASET = "newDataLoader"

CWD = os.getcwd()
DATASETS = os.path.join(CWD, DATASETS)
DATASET = os.path.join(DATASETS, DATASET)

# Checkpoint location:
CHECKPOINT_DIR = 'training_checkpoints_pytorch'
CHECKPOINT_DIR = os.path.join(DATASET, CHECKPOINT_DIR)
CHECKPOINT_PREFIX = 'my_ckpt.pth'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PREFIX)

DATASET_PREFIX_PRETTY = os.path.join(DATASET, "dataset_PRETTY.pkl")

def checkDirs():
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

