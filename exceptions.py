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
