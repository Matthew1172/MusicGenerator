import os.path
from flask import Flask, request, send_file, jsonify
from Generation import *

app = Flask(__name__)

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
    if request.method == 'POST':
        content = request.json
        DATASETS = "datasets"
        content['dataset'] = os.path.join(DATASETS, content['dataset'])
        g = Generation(**content)
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
        g.generate()
        g.save()
        midi = g.GENERATION_PREFIX+"_1.mid"
        mxl = g.GENERATION_PREFIX+"_1.mxl"
        if os.path.exists(mxl) and os.path.exists(midi):
            return jsonify({
                'mxl': mxl.split('\\')[-2:],
                'midi': midi.split('\\')[-2:]
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
