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

test_abc = """X:1
T:Down the Hill
V:1 name=treble
R:air
H:Originally in Gdor and notated in 6/8 time.
H:Version 1 from ONeills. Version 2 from Petrie Collection
H:Related to The Blooming Meadows single jig#23
Z:id:hn-air-1
M:3/4
L:1/8
Q:1/4=160
K:Am
BAG | E2A2A2 | A3EAB | cBABcA | BAGABG | AGEDEF | G4A2 |
B2c2A2 | G2E2D2 | E2A2A2 | A4Bc | BAGABc | d2B2d2 |
efg2e2 | d2B2e2 | A3BAG | A3 :|
|: Bcd | e2a2a2 | a3eab | c\'babc'a | b2g2e2 | d2g2g2 | g4a2 |
bagabg | a2g2ed | e2a2a2 | a3eab | c\'babc'a | bagabg |
a2g2e2 | d2B2e2 |1 A3BAG |A3 :|2 A3BAG | A2^c2e2 ||
|: a2e2^c2 | A2^ceA^c | eA^ce^ce | g2d2B2 | G3BdB |
G3AB=c | d2c2B2 | cBABcA | dcBcdB | e2a2^g2 | a3bc\'b |
a2=g2e2 | d2B2e2 |1 A3BAG | A2^c2e2 :|2 A3BAG | A3 || z ||
P:version 2
AG | E2A2A2 | A4B2 | cBABcA | BAGABG | AGEDEF | G4A2 |
B2c2A2 | G2E2D2 | E2A2A2 | A4c2 | BAGABc | dcBcdB |
e2f2g2 | d2B2e2 | A6- | A4 ||
cd | e2a2a2 | a4b2 | c'babc'a | b2g2e2 | d2g2g2 | g4a2 |
b2c\'2a2 | g2e2d2 | e2a2a2 | a4b2 | c\'babc\'a | bagabg |
a2g2e2 | d2B2^G2 | A6- | A4 ||
a^g | a2e2c2 | A2c2e2 | a2e2c2 | A2c2e2 | g2d2B2 | G2B2d2 |
g2d2B2 | G2B2d2 | cBABcA | dcBcdB | e2a2^g2 | a2b2c'2 |
a2g2e2 | d2B2e2 | A6- | A4 ||"""
g.loadDataFromAbc(test_abc)

g.generate()
g.save()