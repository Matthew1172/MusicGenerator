from music21 import *

def logProcessSlow(position, length, output, fn):
    print("{}/{}\n\n{}\n\n".format(position, length, fn))

def logProcessFast(position, length):
    print("{}/{}\n\n".format(position, length))

def parseAbcString(abc_song):
    print(abc_song)
    pretty_song = []
    try:
        s = converter.parse(abc_song)
        print(s[0].title)
        for m in s[1].elements:
            if isinstance(m, stream.Measure):
                pretty_song.append("|")
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
                    elif isinstance(n, chord.Chord):
                        da += "Chord"
                        da += " "
                        da += "{"
                        for temp in n.notes:
                            da += "Note"
                            da += " "
                            da += temp.nameWithOctave
                            da += " "
                            da += str(temp.quarterLength)
                            da += "$"
                        da += "} "
                        da += "{"
                        da += str(n.quarterLength)
                        da += "}"
                    else:
                        continue
                    pretty_song.append(da)
            elif isinstance(m, spanner.RepeatBracket):
                continue
            else:
                continue
    except:
        pass
    finally:
        return pretty_song[1:]
