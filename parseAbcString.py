from music21 import *

def logProcessSlow(position, length, output, fn):
    print("{}/{}\n\n{}\n\n".format(position, length, fn))

def logProcessFast(position, length, output):
    print("{}/{}\n\n".format(position, length))

def parseToken(t):
    da = ""
    if isinstance(t, note.Note):
        da += "Note"
        da += " "
        da += t.nameWithOctave
        da += " "
        da += str(t.quarterLength)
    elif isinstance(t, note.Rest):
        da += "Rest"
        da += " "
        da += t.name
        da += " "
        da += str(t.quarterLength)
    elif isinstance(t, bar.Barline):
        da += "Bar"
        da += " "
        da += t.type
    elif isinstance(t, clef.Clef):
        da += "Clef"
        da += " "
        da += t.sign
    elif isinstance(t, key.KeySignature):
        da += "Key"
        da += " "
        da += str(t.sharps)
    elif isinstance(t, meter.TimeSignature):
        da += "Time"
        da += " "
        da += str(t.numerator)
        da += " "
        da += str(t.denominator)
    elif isinstance(t, chord.Chord):
        da += "Chord"
        da += " "
        da += "{"
        for temp in t.notes:
            da += "Note"
            da += " "
            da += temp.nameWithOctave
            da += " "
            da += str(temp.quarterLength)
            da += "$"
        da += "} "
        da += "{"
        da += str(t.quarterLength)
        da += "}"
    return da

'''
The input is a song in abc notation
The output is a song in our notation
ex.
["Clef G", "Time 3 4", "Key 2", "Note C4 1.0", ...]
'''
def parseAbcString(abc_song):
    print("-"*10)
    print(abc_song)
    pretty_song = []
    first = True
    try:
        s = converter.parse(abc_song)
        #print(s[0].title)
        for m in s[1].elements:
            if isinstance(m, stream.Measure):
                if first: first = False
                else: pretty_song.append("|")
                for n in m:
                    t = parseToken(n)
                    if t: pretty_song.append(t)
            elif isinstance(m, spanner.RepeatBracket):
                #Append something to pretty_song
                continue
            else:
                t = parseToken(m)
                if t: pretty_song.append(t)
    except:
        pass
    finally:
        return pretty_song
