from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
app.secret_key = 'foot_measure_secret_2026'

# Known sizes in cm
REFERENCE_SIZES = {
    'hand': 8.5,
    'card': 8.56,
    'paper': 21.0
}

def is_foot_shape(contour):
    """Check if contour shape is a foot (long and narrow)"""
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h / w if w > 0 else 0
    area = cv2.contourArea(contour)
    
    # Foot characteristics:
    # - Aspect ratio between 2.2 and 4.0 (longer than wide)
    # - Minimum area to avoid small noise
    if 2.2 < aspect_ratio < 4.0 and area > 5000:
        return True
    return False

def is_reference_object(contour):
    """Check if contour could be a hand, card, or paper"""
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h / w if w > 0 else 0
    area = cv2.contourArea(contour)
    
    # Reference objects are wider (aspect ratio 1.0 to 2.0)
    if 1.0 < aspect_ratio < 2.2 and area > 2000:
        return True
    return False

def detect_foot_and_reference(image_data):
    try:
        img_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None, None
        
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        foot = None
        reference = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Ignore tiny objects
                continue
                
            if is_foot_shape(contour):
                x, y, w, h = cv2.boundingRect(contour)
                foot = {'x': x, 'y': y, 'w': w, 'h': h, 'area': area}
            elif is_reference_object(contour):
                x, y, w, h = cv2.boundingRect(contour)
                reference = {'x': x, 'y': y, 'w': w, 'h': h, 'area': area}
        
        return foot, reference, None
    except Exception as e:
        return None, None, str(e)

def calculate_foot_size(foot_obj, ref_obj, ref_type):
    if not foot_obj or not ref_obj:
        return None
    
    ref_real_cm = REFERENCE_SIZES.get(ref_type, 8.5)
    ref_pixels = ref_obj['w']
    foot_pixels = foot_obj['h']
    
    if ref_pixels == 0:
        return None
    
    cm_per_pixel = ref_real_cm / ref_pixels
    foot_cm = foot_pixels * cm_per_pixel
    
    if foot_cm < 10 or foot_cm > 40:
        return None
    
    return round(foot_cm, 1)

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
    return render_template('measure.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        image = data.get('image')
        ref_type = data.get('reference', 'hand')
        
        foot, reference, error = detect_foot_and_reference(image)
        
        # Case 1: No foot detected
        if not foot:
            return jsonify({
                'success': False, 
                'error': 'foot_not_found',
                'message': '❌ FOOT NOT FOUND! Please place your foot clearly in the frame.'
            })
        
        # Case 2: Foot detected but no reference
        if foot and not reference:
            return jsonify({
                'success': False,
                'error': 'reference_not_found', 
                'message': '❌ REFERENCE NOT FOUND! Please place your HAND (or card/paper) next to your foot.'
            })
        
        # Case 3: Both detected - calculate size
        if foot and reference:
            foot_cm = calculate_foot_size(foot, reference, ref_type)
            
            if foot_cm:
                sizes = get_shoe_sizes(foot_cm)
                return jsonify({
                    'success': True,
                    'foot_cm': foot_cm,
                    'sizes': sizes
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'calculation_error',
                    'message': '⚠️ Could not calculate size. Please ensure foot and reference are side by side.'
                })
        
        return jsonify({
            'success': False,
            'error': 'unknown',
            'message': '⚠️ Please place your FOOT and REFERENCE object clearly in frame.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': 'exception', 'message': str(e)})

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
