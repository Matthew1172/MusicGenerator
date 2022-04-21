import argparse
from Generation import *
from common import DATASETS
from parseAbcString import *

parser = argparse.ArgumentParser(description='Music Generator by Matthew Pecko')
parser.add_argument('--dataset', type=str, default="V1",
                    help='dataset to use')
parser.add_argument('--temperature', type=float, default=0.85,
                    help='temperature - higher will increase diversity')
parser.add_argument('--length', type=int, default=100,
                    help='Length of song to be generated')
parser.add_argument('--songs', type=int, default=3,
                    help='Number of songs to generate')
try:
    parser.add_argument('--random-clef', action=argparse.BooleanOptionalAction,
                        help='Assign a random clef')
    parser.add_argument('--random-key', action=argparse.BooleanOptionalAction,
                        help='Assign a random key signature')
    parser.add_argument('--random-time', action=argparse.BooleanOptionalAction,
                        help='Assign a random time signature')
    parser.add_argument('--random-seq', action=argparse.BooleanOptionalAction,
                        help='Assign a random sequence of notes')
except AttributeError:
    parser.add_argument('--random-clef', default=False, action='store_true',
                        help='Assign a random clef')
    parser.add_argument('--no-random-clef', dest='random_clef', action='store_false',
                        help='Assign a random clef')

    parser.add_argument('--random-key', default=False, action='store_true',
                        help='Assign a random key signature')
    parser.add_argument('--no-random-key', dest='random_key', action='store_false',
                        help='Assign a random key signature')

    parser.add_argument('--random-time', default=False, action='store_true',
                        help='Assign a random time signature')
    parser.add_argument('--no-random-time', dest='random_time', action='store_false',
                        help='Assign a random time signature')

    parser.add_argument('--random-seq', default=False, action='store_true',
                        help='Assign a random sequence of notes')
    parser.add_argument('--no-random-seq', dest='random_seq', action='store_false',
                        help='Assign a random sequence of notes')
except:
    exit(1)

parser.add_argument('--random-seq-length', type=int, default=1,
                    help='Number of random notes to create')
parser.add_argument('--input-clef', type=str, default="Clef G",
                    help='Assign a clef ("Clef G", "Clef F"). Only supports treble clef as of version 1.0')
parser.add_argument('--input-time', type=str, default="Time 4 4",
                    help='Assign a time signature ("Time 4 4", "Time 2 4", "Time 3 4", "Time 27 16")')
parser.add_argument('--input-key', type=str, default="Key 2",
                    help='Assign a key signature ("Key 2", "Key 1", "Key -6", "Key 8")')
parser.add_argument('--input-seq', type=str, default="Note C 1.0",
                    help='Assign a sequence of notes. Use $ as delimiter. ("Note C 1.0", "Note C 1.0$Note F#4 1/3$Note C4 0.5")')
args = parser.parse_args()

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater than 1e-3.")

if args.songs < 1 or args.songs > 100:
    parser.error("--songs has to be greater than 0 and less than 100.")

if args.length < 1 or args.length > 1000:
    parser.error("--length has to be greater than 0 and less than 1000.")

if args.random_seq and (args.random_seq_length < 1 or args.random_seq_length > 1000 or args.random_seq_length > args.length):
    parser.error("if --random-seq is true then --random-seq-length must be less than --length, and has to be greater than 0 and less than 1000.")


g = Generation(dataset=os.path.join(DATASETS, args.dataset),
               input_clef=args.input_clef,
               input_key=args.input_key,
               input_seq=args.input_seq,
               input_time=args.input_time,
               length=args.length,
               random_clef=args.random_clef,
               random_key=args.random_key,
               random_seq=args.random_seq,
               random_seq_length=args.random_seq_length,
               random_time=args.random_time,
               songs=args.songs,
               temperature=args.temperature)

g.loadModel()
g.loadDictionary()
g.setInitClef()
g.setInitKey()
g.setInitTime()
g.setInitSeq()
g.checkInitClef()
g.checkInitKey()
g.checkInitTime()
g.checkInitSeq()

test_abc = "M:6/8\nV:1 name=?\nK:Am\n?"
g.loadDataFromAbc(test_abc)

g.generate()
g.save()