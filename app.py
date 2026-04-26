from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
from datetime import datetime
import base64
import json

app = Flask(__name__)
app.secret_key = 'foot_measure_secret_2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Shoe size charts
FOOT_SIZES = {
    'infant': {'us': {9:0, 9.5:0.5, 10:1, 10.5:1.5, 11:2, 11.5:2.5, 12:3}, 'uk': {9:0, 9.5:0.5, 10:1, 10.5:1.5, 11:2, 11.5:2.5, 12:3}, 'eu': {9:16, 9.5:17, 10:17, 10.5:18, 11:18, 11.5:19, 12:19}},
    'toddler': {'us': {12:4, 12.5:4.5, 13:5, 13.5:5.5, 14:6, 14.5:6.5, 15:7}, 'uk': {12:3, 12.5:3.5, 13:4, 13.5:4.5, 14:5, 14.5:5.5, 15:6}, 'eu': {12:20, 12.5:21, 13:21, 13.5:22, 14:23, 14.5:23, 15:24}},
    'child': {'us': {15:7.5, 16:8, 17:9, 18:10, 19:11, 20:12, 21:13}, 'uk': {15:6.5, 16:7, 17:8, 18:9, 19:10, 20:11, 21:12}, 'eu': {15:25, 16:26, 17:27, 18:28, 19:29, 20:31, 21:32}},
    'teen': {'us': {21:13, 22:1, 23:2, 24:3, 25:4, 26:5, 27:6}, 'uk': {21:12, 22:13, 23:1, 24:2, 25:3, 26:4, 27:5}, 'eu': {21:32, 22:33, 23:34, 24:35, 25:36, 26:37, 27:38}},
    'adult': {'us_men': {22:4, 23:5, 24:6, 25:7, 26:8, 27:9, 28:10, 29:11, 30:12, 31:13}, 'us_women': {21:4, 22:5, 23:6, 24:7, 25:8, 26:9, 27:10, 28:11, 29:12}, 'uk': {22:3.5, 23:4.5, 24:5.5, 25:6.5, 26:7.5, 27:8.5, 28:9.5, 29:10.5}, 'eu': {21:34, 22:35, 23:36, 24:37, 25:38, 26:39, 27:40, 28:41, 29:42, 30:43}}
}

# Product recommendations with affiliate links
PRODUCTS = {
    'infant': [
        {'name': 'Stride Rite Soft Motion', 'price': '$45', 'link': 'https://amazon.com/dp/B08XXX', 'image': '👶'},
        {'name': 'Robeez Baby Shoes', 'price': '$30', 'link': 'https://amazon.com/dp/B09XXX', 'image': '👟'},
        {'name': 'New Balance Kids', 'price': '$40', 'link': 'https://amazon.com/dp/B07XXX', 'image': '👟'}
    ],
    'toddler': [
        {'name': 'Crocs Toddler Clogs', 'price': '$35', 'link': 'https://amazon.com/dp/B06XXX', 'image': '👡'},
        {'name': 'Nike Dynamo Free', 'price': '$55', 'link': 'https://amazon.com/dp/B08YYY', 'image': '👟'},
        {'name': 'Skechers Twinkle Toes', 'price': '$48', 'link': 'https://amazon.com/dp/B07ZZZ', 'image': '✨'}
    ],
    'child': [
        {'name': 'Under Armour Pre-School', 'price': '$50', 'link': 'https://amazon.com/dp/B08AAA', 'image': '🏃'},
        {'name': 'Adidas Cloudfoam', 'price': '$60', 'link': 'https://amazon.com/dp/B09BBB', 'image': '☁️'},
        {'name': 'Puma Kids', 'price': '$45', 'link': 'https://amazon.com/dp/B07CCC', 'image': '🐾'}
    ],
    'adult': [
        {'name': 'Nike Air Max', 'price': '$120', 'link': 'https://amazon.com/dp/B08DDD', 'image': '👟'},
        {'name': 'Adidas Ultraboost', 'price': '$180', 'link': 'https://amazon.com/dp/B09EEE', 'image': '🏃'},
        {'name': 'New Balance 990', 'price': '$150', 'link': 'https://amazon.com/dp/B07FFF', 'image': '🦶'}
    ]
}

def detect_foot_pixels(image_data):
    try:
        if not image_data or ',' not in image_data:
            return None
        img_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            foot_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(foot_contour) > 1000:
                x, y, w, h = cv2.boundingRect(foot_contour)
                return {'length': h, 'width': w, 'x': x, 'y': y}
        return None
    except Exception as e:
        print(f"Detection error: {e}")
        return None

def calculate_foot_size(pixel_length, frame_height, age_category):
    scaling_factors = {'infant': 0.08, 'toddler': 0.10, 'child': 0.12, 'teen': 0.14, 'adult': 0.16}
    scale = scaling_factors.get(age_category, 0.12)
    return round(pixel_length * scale, 1)

def get_shoe_sizes(foot_cm, age_category, gender='unisex'):
    sizes = {}
    try:
        if age_category == 'adult':
            us_chart = FOOT_SIZES['adult']['us_men'] if gender == 'male' else FOOT_SIZES['adult']['us_women']
            uk_chart = FOOT_SIZES['adult']['uk']
            eu_chart = FOOT_SIZES['adult']['eu']
        else:
            us_chart = FOOT_SIZES[age_category]['us']
            uk_chart = FOOT_SIZES[age_category]['uk']
            eu_chart = FOOT_SIZES[age_category]['eu']
        closest_us = min(us_chart.keys(), key=lambda x: abs(x - foot_cm))
        sizes['US'] = us_chart[closest_us]
        closest_uk = min(uk_chart.keys(), key=lambda x: abs(x - foot_cm))
        sizes['UK'] = uk_chart[closest_uk]
        closest_eu = min(eu_chart.keys(), key=lambda x: abs(x - foot_cm))
        sizes['EU'] = eu_chart[closest_eu]
    except:
        sizes = {'US': '--', 'UK': '--', 'EU': '--'}
    return sizes

def get_growth_prediction(us_size, age_category):
    if age_category in ['infant', 'toddler', 'child', 'teen']:
        return {
            '3_months': round(us_size + 0.5, 1),
            '6_months': round(us_size + 1, 1),
            '12_months': round(us_size + 2, 1)
        }
    return None

@app.route('/')
def index():
    return redirect(url_for('info'))

@app.route('/info', methods=['GET', 'POST'])
def info():
    if request.method == 'POST':
        session['user_name'] = request.form.get('name', 'Guest')
        session['user_age'] = int(request.form.get('age', 25))
        session['age_category'] = request.form.get('age_category', 'adult')
        session['gender'] = request.form.get('gender', 'unisex')
        return redirect(url_for('measure'))
    return render_template('info.html')

@app.route('/measure')
def measure():
    return render_template('measure.html', 
                         user_name=session.get('user_name', 'Guest'),
                         age_category=session.get('age_category', 'adult'),
                         gender=session.get('gender', 'unisex'))

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        image = data.get('image')
        frame_height = data.get('frame_height', 480)
        
        foot = detect_foot_pixels(image)
        
        if foot and foot['length'] > 100:
            foot_cm = calculate_foot_size(foot['length'], frame_height, session.get('age_category', 'adult'))
            sizes = get_shoe_sizes(foot_cm, session.get('age_category', 'adult'), session.get('gender', 'unisex'))
            growth = get_growth_prediction(sizes['US'], session.get('age_category', 'adult'))
            products = PRODUCTS.get(session.get('age_category', 'adult'), PRODUCTS['adult'])
            
            return jsonify({
                'success': True,
                'foot_cm': foot_cm,
                'sizes': sizes,
                'growth': growth,
                'products': products,
                'pixel_length': foot['length']
            })
        else:
            return jsonify({'success': False, 'error': 'No foot detected'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('info'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
