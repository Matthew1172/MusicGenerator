'''TODO: create custom exceptions for flask server'''
class NoteNotFoundInDictionary(Exception):
    def __init__(self, notes):
        m = [k for (k, v) in notes.items()]
        msg = "One or more notes was not found in the dictionary: {}".format(m)
        super().__init__(msg)

class TimeNotFoundInDictionary(Exception):
    def __init__(self, time):
        msg = "The time signature {} was not found in the dictionary.".format(time)
        super().__init__(msg)

class ClefNotFoundInDictionary(Exception):
    def __init__(self, clef):
        msg = "The clef {} was not found in the dictionary.".format(clef)
        super().__init__(msg)

class KeyNotFoundInDictionary(Exception):
    def __init__(self, key):
        msg = "The key signature {} was not found in the dictionary.".format(key)
        super().__init__(msg)

class DatasetNotFound(Exception):
    def __init__(self, ds):
        msg = "The dataset {} was not found on the server.".format(ds)
        super().__init__(msg)

class CouldNotSaveInference(Exception):
    def __init__(self, dir):
        msg = "The inference could not be saved in this directory {}".format(dir)
        super().__init__(msg)

class CouldNotSaveMidiFile(Exception):
    def __init__(self, dir):
        msg = "The midi file could not be saved. {}".format(dir)
        super().__init__(msg)

class CouldNotSaveMxlFile(Exception):
    def __init__(self, dir):
        msg = "The mxl file could not be saved. {}".format(dir)
        super().__init__(msg)

class CouldNotSaveTxtFile(Exception):
    def __init__(self, dir):
        msg = "The text file could not be saved. {}".format(dir)
        super().__init__(msg)
