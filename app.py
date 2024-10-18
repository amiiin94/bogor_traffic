from flask import Flask, render_template, Response
from btm import generate_frames
import pandas as pd
from flask import jsonify
import json


app = Flask(__name__)

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
    return render_template('depan_mall_btm.html')

@app.route('/kapten-muslihat')
def kapten_muslihat_view():
    return render_template('kapten_muslihat.html')

@app.route('/video_feed_btm')
def video_feed_btm():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_traffic_data')
def get_traffic_data():
    with open('static/traffic_data.json', 'r') as file:
        data = json.load(file)
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)