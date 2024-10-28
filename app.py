from flask import Flask, render_template, Response
from python.btm import generate_frames
from python.kapten_muslihat import generate_kapten_muslihat_frames
from python.lawanggintung import generate_lawanggintung_frames
from python.tugu_kujang import generate_tugu_kujang_frames
import pandas as pd
from flask import jsonify
import json


app = Flask(__name__)

# Route web
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tugu-kujang')
def tugu_kujang():
    return render_template('tugu_kujang.html')

@app.route('/simpang-baranang-siang')
def simpang_baranang_siang():
    return render_template('simpang_baranang_siang.html')

@app.route('/lawanggintung')
def lawanggintung():
    return render_template('lawanggintung.html')

@app.route('/depan-mall-btm')
def depan_mall_btm():
    return render_template('btm.html')

@app.route('/kapten-muslihat')
def kapten_muslihat_view():
    return render_template('kapten_muslihat.html')

# Route video
@app.route('/video_feed_btm')
def video_feed_btm():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_kapten_muslihat')
def video_kapten_muslihat():
    return Response(generate_kapten_muslihat_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_lawanggintung')
def video_lawanggintung():
    return Response(generate_lawanggintung_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_tugu_kujang')
def video_tugu_kujang():
    return Response(generate_tugu_kujang_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Route file json
@app.route('/btm_data')
def btm_data():
    with open('static/json/btm_data.json', 'r') as file:
        data = json.load(file)
    return jsonify(data)

@app.route('/kapten_muslihat_data_1')
def kapten_muslihat_data_1():
    with open('static/json/kapten_muslihat_1.json', 'r') as file:
        data = json.load(file)
    return jsonify(data)

@app.route('/kapten_muslihat_data_2')
def kapten_muslihat_data_2():
    with open('static/json/kapten_muslihat_2.json', 'r') as file:
        data = json.load(file)
    return jsonify(data)

@app.route('/lawanggintung_data')
def lawanggintung_data():
    with open('static/json/lawanggintung_data.json', 'r') as file:
        data = json.load(file)
    return jsonify(data)

@app.route('/tugu_kujang_data_1')
def tugu_kujang_data_1():
    with open('static/json/tugu_kujang_data_1.json', 'r') as file:
        data = json.load(file)
    return jsonify(data)

@app.route('/tugu_kujang_data_2')
def tugu_kujang_data_2():
    with open('static/json/tugu_kujang_data_2.json', 'r') as file:
        data = json.load(file)
    return jsonify(data)









if __name__ == '__main__':
    app.run(debug=True)