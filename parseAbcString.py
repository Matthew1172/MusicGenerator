from music21 import *
from Dictionary import *
from random import randint

def logProcessSlow(position, length, output, fn):
    print("{}/{}\n\n{}\n\n".format(position, length, fn))

def logProcessFast(position, length, output):
    print("{}/{}\n\n".format(position, length))

def dontCare(t):
    if "X:" in t or "T:" in t:
        return True
    else:
        return False

def parseAbcString(abc_song):
    print(abc_song)
    pretty_song = []
    a = abcFormat.ABCHandler()
    try:
        a.process(abc_song)
        tokenColls = a.splitByVoice()
        for i in range(len(tokenColls)):
            #ignore reel and title headers. Some tokens are busted? ex. '(3', '[1', '[2', '[3', '|]'
            src = [t.src for t in tokenColls[i].tokens if not dontCare(t.src)]
            pretty_song += src
    except:
        pass
    finally:
        return pretty_song
