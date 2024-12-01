# backend/app.py

import os
import time
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from utils import (
    function_Prewitt,
    function_Sobel,
    function_Kirsch,
    function_wavelet,
)
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folders
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOTS_FOLDER'] = PLOTS_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict/<method>', methods=['POST'])
def predict(method):
    if method not in ['prewitt', 'sobel', 'kirsch', 'wavelet']:
        return jsonify({'error': 'Invalid method'}), 400

    if 'original_video' not in request.files or 'upconverted_video' not in request.files:
        return jsonify({'error': 'Both original and upconverted videos are required'}), 400

    original_video = request.files['original_video']
    upconverted_video = request.files['upconverted_video']

    if original_video.filename == '' or upconverted_video.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if original_video and allowed_file(original_video.filename) and upconverted_video and allowed_file(upconverted_video.filename):
        original_filename = secure_filename(original_video.filename)
        upconverted_filename = secure_filename(upconverted_video.filename)

        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        upconverted_path = os.path.join(app.config['UPLOAD_FOLDER'], upconverted_filename)

        original_video.save(original_path)
        upconverted_video.save(upconverted_path)

        # Start processing
        start_time = time.time()

        # Retrieve additional parameters
        edge_threshold = float(request.form.get('edge_threshold', 100))
        binary_classification = True
        n_frames = int(request.form.get('n_frames', 10))

        try:
            if method == 'prewitt':
                function_Prewitt(original_path, upconverted_path, edge_threshold, binary_classification, n_frames)                
            elif method == 'sobel':
                function_Sobel(original_path, upconverted_path, edge_threshold, binary_classification, n_frames)
            elif method == 'kirsch':
                function_Kirsch(original_path, upconverted_path, edge_threshold, binary_classification, n_frames)
            elif method == 'wavelet':
                function_wavelet(original_path, upconverted_path, edge_threshold, binary_classification, n_frames)
            plot_filename = 'edge_intensity.png'  # Assumed naming convention
            classification_plot = 'classification.png'
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        os.remove(original_path)
        os.remove(upconverted_path)
        end_time = time.time()
        processing_time = end_time - start_time

        # Construct plot URLs
        plot_url = f"/static/plots/{plot_filename}"
        classification_plot_url = f"/static/plots/{classification_plot}"
        # Prepare response
        response = {
            'processing_time': round(processing_time, 2),
            'plot_url': plot_url,
            'classification_plot_url': classification_plot_url
        }

        return jsonify(response), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/static/plots/<filename>')
def send_plot(filename):
    return send_from_directory(app.config['PLOTS_FOLDER'], filename)

@app.route('/')
def index():
    # Serve the frontend
    # return "hahah"
    return send_from_directory('static/templates', 'index.html')

if __name__ == '__main__':
    app.run(port=5000,debug=True)