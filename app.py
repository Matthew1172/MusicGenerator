import os.path
from flask import Flask, request, send_file, jsonify, make_response
from Generation import *
from music21 import *

app = Flask(__name__)

# INITIALIZATION
xml_response_headers = {"Content-Type": "text/xml",
                        "charset":      "utf-8"
                        }

#use this parameter or extract it from the metadata somehow
timesignature = meter.TimeSignature('4/4')

'''
request looks like:
req = {
    dataset: "Good"
    input_clef: "Clef G"
    input_key: "Key 2"
    input_seq: "Note C 1.0"
    input_time: "Time 4 4"
    length: 100
    random_clef: False
    random_key: False
    random_seq: False
    random_seq_length: 1
    random_time: False
    songs: 1
    temperature: 0.85
}
'''
@app.route('/predict', methods=['POST'])
def predict():
    print("req: ", request)
    if request.method == 'POST':
        print("req: ", request)
        content = request.json

        '''TODO: check all keys of content and do error handling.'''
        print("The content of the post req: ", content)

        DATASETS = "datasets"
        content['dataset'] = os.path.join(DATASETS, content['dataset'])
        g = Generation(**content)
        g.loadModel()
        g.loadDictionary()
        g.setInitClef()
        g.setInitKey()
        g.setInitTime()
        try:
            g.setInitSeq()
        except:
            return jsonify({'error': "note not found"})
        g.checkInitClef()
        g.checkInitKey()
        g.checkInitTime()
        g.checkInitSeq()
        g.generate()
        g.save()
        midi = g.GENERATION_PREFIX+"_1.mid"
        mxl = g.GENERATION_PREFIX+"_1.mxl"
        if os.path.exists(mxl) and os.path.exists(midi):
            mxl_path = mxl.split('\\')[-2:]
            midi_path = midi.split('\\')[-2:]
            if len(mxl_path) < 2:
                mxl_path = mxl.split('/')[-2:]
                midi_path = midi.split('/')[-2:]
            return jsonify({
                'mxl': mxl_path,
                'midi': midi_path
            })
        else:
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



@app.route('/test-generate', methods=['GET'])
def ex():
    folder = request.args.get('folder')
    file = request.args.get('file')
    path = os.path.join(os.getcwd(), os.path.join("outputs", os.path.join(folder, file)))
    print(path)
    _current_sheet = converter.parse(path)
    return sheet_to_xml_response(_current_sheet)


def insert_musicxml_metadata(sheet: stream.Stream):
    """
    Insert various metadata into the provided XML document
    The timesignature in particular is required for proper MIDI conversion
    """
    global timesignature

    from music21.clef import TrebleClef, BassClef, Treble8vbClef
    for part, name, clef in zip(
            sheet.parts,
            ['soprano', 'alto', 'tenor', 'bass'],
            [TrebleClef(), TrebleClef(), Treble8vbClef(), BassClef()]
    ):
        # empty_part = part.template()
        part.insert(0, timesignature)
        part.insert(0, clef)
        part.id = name
        part.partName = name

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
