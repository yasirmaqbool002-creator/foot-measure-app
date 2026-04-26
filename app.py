from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
app.secret_key = 'foot_measure_secret_2026'

def is_foot_shape(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h / w if w > 0 else 0
    area = cv2.contourArea(contour)
    
    if 2.2 < aspect_ratio < 4.0 and area > 5000:
        return True
    return False

def detect_foot_only(image_data):
    try:
        img_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:
                continue
                
            if is_foot_shape(contour):
                x, y, w, h = cv2.boundingRect(contour)
                return {'x': x, 'y': y, 'w': w, 'h': h, 'area': area}, img.shape[0]
        
        return None, None
    except Exception as e:
        print(f"Detection error: {e}")
        return None, None

def calculate_foot_size(pixel_length, frame_height):
    ratio = pixel_length / frame_height
    
    if ratio < 0.3:
        return 18
    elif ratio < 0.5:
        return 22
    elif ratio < 0.7:
        return 25
    elif ratio < 0.9:
        return 28
    else:
        return 31

def get_shoe_sizes(foot_cm):
    if foot_cm <= 20:
        us = 2
    elif foot_cm <= 21:
        us = 3
    elif foot_cm <= 22:
        us = 4
    elif foot_cm <= 23:
        us = 5
    elif foot_cm <= 24:
        us = 6
    elif foot_cm <= 25:
        us = 7
    elif foot_cm <= 26:
        us = 8
    elif foot_cm <= 27:
        us = 9
    elif foot_cm <= 28:
        us = 10
    elif foot_cm <= 29:
        us = 11
    elif foot_cm <= 30:
        us = 12
    else:
        us = 13
    
    return {'US': us, 'UK': us - 1, 'EU': us + 32}

@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/info', methods=['GET', 'POST'])
def info():
    if request.method == 'POST':
        session['user_name'] = request.form.get('name', 'Guest')
        session['age_category'] = request.form.get('age_category', 'adult')
        session['gender'] = request.form.get('gender', 'unisex')
        return redirect(url_for('measure'))
    return render_template('info.html')

@app.route('/measure')
def measure():
    return render_template('measure.html', user_name=session.get('user_name', 'Guest'))

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        image = data.get('image')
        frame_height = data.get('frame_height', 480)
        
        foot, height = detect_foot_only(image)
        
        if not foot:
            return jsonify({
                'success': False,
                'error': 'foot_not_found',
                'message': '❌ FOOT NOT FOUND! Please place your foot clearly.'
            })
        
        foot_cm = calculate_foot_size(foot['h'], frame_height)
        sizes = get_shoe_sizes(foot_cm)
        
        return jsonify({
            'success': True,
            'foot_cm': foot_cm,
            'sizes': sizes,
            'pixel_length': foot['h'],
            'user_name': session.get('user_name', 'Guest')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': 'exception', 'message': str(e)})

@app.route('/get_user')
def get_user():
    return jsonify({'name': session.get('user_name', 'Guest')})

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
