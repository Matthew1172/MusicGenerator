import os
#from IPython import display as ipythondisplay
import regex as re
import time

#current directory
cwd = os.getcwd()
#collection of all outputs directory
opd = os.path.join(cwd, "outputs")
#output directory for this generation

op = os.path.join(opd, "output@"+time.asctime().replace(' ', '').replace(':', ''))

#create directories
try:
    os.mkdir(opd)
except FileExistsError:
    print("The outputs directory already exists...")

try:
    os.mkdir(op)
except FileExistsError:
    print("The directory {} already exists...".format(op))

def save_song_to_abc(song, filename="sampleFromDataset"):
    save_name = os.path.join(op, "{}.abc".format(filename))
    with open(save_name, "w") as f:
        f.write(song)
    return filename

def abc2wav(abc_file):
    suf = abc_file.rstrip('.abc')
    print(abc_file)
    cmd = "abc2midi '{}' -o '{}'".format(abc_file, suf + ".mid")
    os.system(cmd)
    #can't write to file.. tmp.mid is not a midi file. We will just save the wav file as the name of the mid file
    #cmd = "timidity {}.mid -Ow {}.wav".format(suf, suf)
    cmd = "timidity '{}'.mid -Ow".format(suf, suf)
    return os.system(cmd)

def play_wav(wav_file):
    #return ipythondisplay.Audio(wav_file)
    pass

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