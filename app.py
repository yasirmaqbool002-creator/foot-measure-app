from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
from datetime import datetime
import base64
import json
import qrcode
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = Flask(__name__)
app.secret_key = 'foot_measure_secret_2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Shoe size charts
FOOT_SIZES = {
    'infant': {'us': {9:0, 9.5:0.5, 10:1, 10.5:1.5, 11:2, 11.5:2.5, 12:3}, 'uk': {9:0, 9.5:0.5, 10:1, 10.5:1.5, 11:2, 11.5:2.5, 12:3}, 'eu': {9:16, 9.5:17, 10:17, 10.5:18, 11:18, 11.5:19, 12:19}},
    'toddler': {'us': {12:4, 12.5:4.5, 13:5, 13.5:5.5, 14:6, 14.5:6.5, 15:7}, 'uk': {12:3, 12.5:3.5, 13:4, 13.5:4.5, 14:5, 14.5:5.5, 15:6}, 'eu': {12:20, 12.5:21, 13:21, 13.5:22, 14:23, 14.5:23, 15:24}},
    'child': {'us': {15:7.5, 16:8, 17:9, 18:10, 19:11, 20:12, 21:13}, 'uk': {15:6.5, 16:7, 17:8, 18:9, 19:10, 20:11, 21:12}, 'eu': {15:25, 16:26, 17:27, 18:28, 19:29, 20:31, 21:32}},
    'teen': {'us': {21:13, 22:1, 23:2, 24:3, 25:4, 26:5, 27:6}, 'uk': {21:12, 22:13, 23:1, 24:2, 25:3, 26:4, 27:5}, 'eu': {21:32, 22:33, 23:34, 24:35, 25:36, 26:37, 27:38}},
    'adult': {'us_men': {22:4, 23:5, 24:6, 25:7, 26:8, 27:9, 28:10, 29:11, 30:12, 31:13}, 'us_women': {21:4, 22:5, 23:6, 24:7, 25:8, 26:9, 27:10, 28:11, 29:12}, 'uk': {22:3.5, 23:4.5, 24:5.5, 25:6.5, 26:7.5, 27:8.5, 28:9.5, 29:10.5}, 'eu': {21:34, 22:35, 23:36, 24:37, 25:38, 26:39, 27:40, 28:41, 29:42, 30:43}}
}

PERCENTILES = {
    'infant': [(6, 10), (9, 25), (11, 50), (12, 75), (14, 90)],
    'toddler': [(12, 10), (13, 25), (14, 50), (15, 75), (16, 90)],
    'child': [(16, 10), (17, 25), (18, 50), (19, 75), (21, 90)],
    'teen': [(21, 10), (23, 25), (25, 50), (27, 75), (29, 90)],
    'adult': [(22, 10), (24, 25), (26, 50), (28, 75), (30, 90)]
}

family_profiles = {}

def detect_foot_pixels(image_data):
    try:
        if not image_data or ',' not in image_data:
            return None
            
        img_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
            
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            foot_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(foot_contour)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(foot_contour)
                return {'length': h, 'width': w, 'x': x, 'y': y, 'area': area}
        return None
    except Exception as e:
        print(f"Detection error: {e}")
        return None

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

def get_percentile(foot_cm, age_category):
    data = PERCENTILES.get(age_category, PERCENTILES['adult'])
    for cm, pct in data:
        if foot_cm <= cm:
            return pct
    return 95

def generate_qr_code(foot_data):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(json.dumps(foot_data))
    qr.make(fit=True)
    img = qr.make_image(fill_color="#764ba2", back_color="white")
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/info', methods=['GET', 'POST'])
def info():
    if request.method == 'POST':
        session['user_name'] = request.form.get('name', 'Guest')
        session['user_age'] = int(request.form.get('age', 25))
        session['age_category'] = request.form.get('age_category', 'adult')
        session['gender'] = request.form.get('gender', 'unisex')
        if session['age_category'] in ['infant', 'toddler', 'child', 'teen']:
            return render_template('growth_tracking.html')
        return redirect(url_for('measure'))
    return render_template('info.html')

@app.route('/growth-tracking', methods=['POST'])
def growth_tracking():
    session['track_growth'] = request.form.get('track_growth') == 'yes'
    session['child_name'] = request.form.get('child_name', session.get('user_name'))
    return redirect(url_for('measure'))

@app.route('/measure')
def measure():
    return render_template('measure.html', 
                         user_name=session.get('user_name', 'Guest'),
                         age_category=session.get('age_category', 'adult'),
                         track_growth=session.get('track_growth', False))

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        image = data.get('image')
        frame_height = data.get('frame_height', 480)
        
        foot = detect_foot_pixels(image)
        
        if foot and foot['length'] > 50:
            foot_cm = calculate_foot_size(foot['length'], frame_height)
            foot_width_cm = round(foot_cm * 0.38, 1)
            sizes = get_shoe_sizes(foot_cm, session.get('age_category', 'adult'), session.get('gender', 'unisex'))
            percentile = get_percentile(foot_cm, session.get('age_category', 'adult'))
            
            result = {
                'success': True,
                'foot_cm': foot_cm,
                'foot_width_cm': foot_width_cm,
                'sizes': sizes,
                'percentile': percentile,
                'pixel_length': foot['length']
            }
            session['last_result'] = result
            return jsonify(result)
        else:
            return jsonify({'success': False, 'error': 'No foot detected', 'pixel_length': 0})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save-profile', methods=['POST'])
def save_profile():
    data = request.json
    family_id = session.get('user_name', 'Guest')
    if family_id not in family_profiles:
        family_profiles[family_id] = []
    family_profiles[family_id].append({
        'name': data.get('name'),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'foot_cm': data.get('foot_cm'),
        'size_us': data.get('size_us')
    })
    return jsonify({'success': True})

@app.route('/get-family')
def get_family():
    family_id = session.get('user_name', 'Guest')
    return jsonify(family_profiles.get(family_id, []))

@app.route('/qr-code')
def qr_code():
    result = session.get('last_result', {})
    qr_data = {
        'size_us': result.get('sizes', {}).get('US'),
        'foot_cm': result.get('foot_cm'),
        'date': datetime.now().strftime('%Y-%m-%d')
    }
    qr_base64 = generate_qr_code(qr_data)
    return jsonify({'qr_image': qr_base64})

@app.route('/download-pdf')
def download_pdf():
    result = session.get('last_result', {})
    filename = f"reports/foot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Foot Measurement Report", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Name: {session.get('user_name', 'Guest')}", styles['Normal']))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Foot Length: {result.get('foot_cm', '--')} cm", styles['Normal']))
    story.append(Paragraph(f"US Size: {result.get('sizes', {}).get('US', '--')}", styles['Normal']))
    doc.build(story)
    return send_file(filename, as_attachment=True)

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
