import os
from IPython import display as ipythondisplay
import regex as re

cwd = os.getcwd()
outputDir = "output"
op = os.path.join(cwd, outputDir)

def save_song_to_abc(song, filename="tmp"):
    save_name = os.path.join(op, "{}.abc".format(filename))
    with open(save_name, "w") as f:
        f.write(song)
    return filename

def abc2wav(abc_file):
    suf = abc_file.rstrip('.abc')
    print(abc_file)
    cmd = "abc2midi {} -o {}".format(abc_file, suf + ".mid")
    os.system(cmd)
    #can't write to file.. tmp.mid is not a midi file. We will just save the wav file as the name of the mid file
    #cmd = "timidity {}.mid -Ow {}.wav".format(suf, suf)
    cmd = "timidity {}.mid -Ow".format(suf, suf)
    return os.system(cmd)

def play_wav(wav_file):
    return ipythondisplay.Audio(wav_file)

def play_song(song):
    basename = os.path.join(op, save_song_to_abc(song))
    ret = abc2wav(basename + '.abc')
    if ret == 0:  # did not suceed
        return play_wav(basename + '.wav')
    return None

def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs
