import os.path
from flask import Flask, request, send_file, jsonify, make_response
from Generation import *
from music21 import *
from exceptions import *
from Dictionary import *

from music21.clef import TrebleClef, BassClef, Treble8vbClef

app = Flask(__name__)

# INITIALIZATION
xml_response_headers = {"Content-Type": "text/xml",
                        "charset":      "utf-8"
                        }

'''TODO: remove this and extract time signature properly'''
#use this parameter or extract it from the metadata somehow
timesignature = meter.TimeSignature('4/4')

'''TODO: add a routes to get all notes, clefs, times, and keys available in dictionary'''
'''TODO: add route to get all available datasets'''
'''
try:
    self.dic.load_dictionary(self.DATASET)
    self.dic.load_list(self.DATASET)
except:
    print(
        "No dictionary file available for loading. Please run the Extraction.py script before generation or training.")
    exit(-998)
'''

@app.route('/datasets', methods=['GET'])
def clefs():
    if request.method == 'GET':
        dic = Dictionary()
        datasets = "datasets"

        dataset_folder_names = [os.path.split(os.path.split(dp)[0])[1] for dp, dn, filenames in os.walk(os.path.join(os.getcwd(), datasets)) for f in filenames if
                      os.path.splitext(f)[1] == '.pth']

        return jsonify({'datasets': dataset_folder_names})

@app.route('/clefs', methods=['GET'])
def clefs():
    if request.method == 'GET':
        dic = Dictionary()
        datasets = "datasets"
        dataset = request.args.get('dataset')
        path = os.path.join(os.getcwd(), os.path.join(datasets, dataset))
        dic.load_list(path)
        return jsonify({'clefs': [i for i in dic.idx2word if "Clef" in i]})

@app.route('/keys', methods=['GET'])
def keys():
    if request.method == 'GET':
        dic = Dictionary()
        datasets = "datasets"
        dataset = request.args.get('dataset')
        path = os.path.join(os.getcwd(), os.path.join(datasets, dataset))
        dic.load_list(path)
        return jsonify({'keys': [i for i in dic.idx2word if "Key" in i]})

@app.route('/times', methods=['GET'])
def times():
    if request.method == 'GET':
        dic = Dictionary()
        datasets = "datasets"
        dataset = request.args.get('dataset')
        path = os.path.join(os.getcwd(), os.path.join(datasets, dataset))
        dic.load_list(path)
        return jsonify({'times': [i for i in dic.idx2word if "Time" in i]})

@app.route('/notes', methods=['GET'])
def notes():
    if request.method == 'GET':
        dic = Dictionary()
        datasets = "datasets"
        dataset = request.args.get('dataset')
        path = os.path.join(os.getcwd(), os.path.join(datasets, dataset))
        dic.load_list(path)
        return jsonify({'notes': [i for i in dic.idx2word if "Note" in i]})

'''
request looks like:
req = {
    dataset: "Good"
    length: 100
    random_seq_length: 1
    songs: 1
    temperature: 0.85
    abc: "M:?\nV:1 name=?\nK:?\n?"
}
'''
@app.route('/mgen', methods=['POST'])
def mgen():
    if request.method == 'POST':
        content = request.json

        '''TODO: check all keys of content and do error handling.'''

        DATASETS = "datasets"
        content['dataset'] = os.path.join(DATASETS, content['dataset'])
        g = Generation(**content)

        try:
            g.checkDataset()
        except DatasetNotFound as dnf:
            return jsonify({'error': str(dnf)})

        g.loadModel()
        g.loadDictionary()
        g.setInitClef()
        g.setInitKey()
        g.setInitTime()
        g.setInitSeq()

        abc = content['abc']


        print(abc)


        g.loadDataFromAbc(abc)

        '''TODO: check for custom exceptions and return proper error codes.
        Append all errors to one error code so the user can see everything at once.'''
        try:
            g.checkInitClef()
            g.checkInitKey()
            g.checkInitTime()
            g.checkInitSeq()
        except NoteNotFoundInDictionary as nnf:
            print(nnf)
            return jsonify({'error': str(nnf)})
        except ClefNotFoundInDictionary as cnf:
            print(cnf)
            return jsonify({'error': str(cnf)})
        except TimeNotFoundInDictionary as tnf:
            print(tnf)
            return jsonify({'error': str(tnf)})
        except KeyNotFoundInDictionary as knf:
            print(knf)
            return jsonify({'error': str(knf)})
        except:
            print("Could not run generation with inputs.")
            return jsonify({'error': "could not run generation with inputs."})


        print(g.iSeq)



        g.generate()

        try:
            g.save()
        except CouldNotSaveInference as e:
            print(e)
            return jsonify({'error': str(e)})
        except CouldNotSaveMidiFile as e:
            print(e)
            return jsonify({'error': str(e)})
        except CouldNotSaveMxlFile as e:
            print(e)
            return jsonify({'error': str(e)})
        except CouldNotSaveTxtFile as e:
            print(e)
            return jsonify({'error': str(e)})


        midi = g.GENERATION_PREFIX+"_1.mid"
        mxl = g.GENERATION_PREFIX+"_1.mxl"
        if os.path.exists(mxl) and os.path.exists(midi):
            mxl_path = mxl.split('\\')[-2:]
            midi_path = midi.split('\\')[-2:]
            if len(mxl_path) < 2:
                mxl_path = mxl.split('/')[-2:]
                midi_path = midi.split('/')[-2:]


            print("mxl: {}\nmidi: {}".format(mxl_path, midi_path))


            return jsonify({
                'mxl': mxl_path,
                'midi': midi_path
            })
        else:


            print("mxl or midi file does not exist.")


            return jsonify({'saved': False})

@app.route('/mxl', methods=['GET'])
def mxl():
    if request.method == 'GET':
        folder = request.args.get('folder')
        file = request.args.get('file')
        path = os.path.join(os.getcwd(), os.path.join("outputs", os.path.join(folder, file)))
        return send_file(path,
                         mimetype='application/vnd.recordare.musicxml',
                         attachment_filename=file,
                         as_attachment=True)

@app.route('/midi', methods=['GET'])
def midi():
    if request.method == 'GET':
        folder = request.args.get('folder')
        file = request.args.get('file')
        path = os.path.join(os.getcwd(), os.path.join("outputs", os.path.join(folder, file)))
        return send_file(path, mimetype='audio/midi')

@app.route('/mxl-data', methods=['GET'])
def ex():
    folder = request.args.get('folder')
    file = request.args.get('file')
    path = os.path.join(os.getcwd(), os.path.join("outputs", os.path.join(folder, file)))


    print(path)


    _current_sheet = converter.parse(path)
    #next line throws error
    return sheet_to_xml_response(_current_sheet)


def insert_musicxml_metadata(sheet: stream.Stream):
    """
    Insert various metadata into the provided XML document
    The timesignature in particular is required for proper MIDI conversion
    """
    md = metadata.Metadata()
    sheet.insert(0, md)

    # required for proper musicXML formatting
    sheet.metadata.title = 'mgen'
    sheet.metadata.composer = 'mgen'

def sheet_to_xml_bytes(sheet: stream.Stream):
    """Convert a music21 sheet to a MusicXML document"""
    # first insert necessary MusicXML metadata
    insert_musicxml_metadata(sheet)

    sheet_to_xml_bytes = musicxml.m21ToXml.GeneralObjectExporter(sheet).parse()

    return sheet_to_xml_bytes

def sheet_to_xml_response(sheet: stream.Stream):
    """Generate and send XML sheet"""
    xml_sheet_bytes = sheet_to_xml_bytes(sheet)

    response = make_response((xml_sheet_bytes, xml_response_headers))
    return response
