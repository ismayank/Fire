from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit
import os
import cv2
import numpy as np
import json
import threading
import time
from datetime import datetime
import base64
from werkzeug.utils import secure_filename
import tempfile
import shutil
from webcam_handler import WebcamHandler

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fire_detection_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize webcam handler
webcam_handler = WebcamHandler(socketio)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['STATIC_FOLDER'] = 'static'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Global variables for processing status
processing_status = {
    'is_processing': False,
    'current_file': None,
    'progress': 0,
    'detections': [],
    'error': None
}

# Preloaded videos with metadata
PRELOADED_VIDEOS = [
    {
        'name': 'Car Fire Detection',
        'filename': 'car-fire.mp4',
        'description': 'Fire detection in a car accident scenario',
        'thumbnail': 'car-fire-thumb.jpg'
    },
    {
        'name': 'Bike Fire Detection', 
        'filename': 'bike-fire.mp4',
        'description': 'Fire detection involving a motorcycle',
        'thumbnail': 'bike-fire-thumb.jpg'
    },
    {
        'name': 'Building Fire',
        'filename': 'fire3.mp4',
        'description': 'Building fire detection scenario',
        'thumbnail': 'fire3-thumb.jpg'
    },
    {
        'name': 'Vehicle Crash Fire',
        'filename': 'car-fire-2.mp4',
        'description': 'Vehicle crash with subsequent fire',
        'thumbnail': 'car-fire-2-thumb.jpg'
    }
]

class FireDetector:
    def __init__(self):
        self.fire_cascade = cv2.CascadeClassifier()
        # Load the pre-trained fire detection model if available
        try:
            self.fire_cascade.load('fire_detection.xml')
        except:
            print("Fire cascade classifier not found, using HSV-based detection")
            self.fire_cascade = None
    
    def detect_fire_hsv(self, frame):
        """HSV-based fire detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for fire color (orange-red)
        lower_fire = np.array([0, 50, 50])
        upper_fire = np.array([35, 255, 255])
        
        mask = cv2.inRange(hsv, lower_fire, upper_fire)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(1.0, area / 10000)  # Simple confidence calculation
                detections.append({
                    'box': [x, y, x+w, y+h],
                    'confidence': confidence,
                    'method': 'hsv'
                })
        
        return detections
    
    def process_frame(self, frame):
        """Process a single frame for fire detection"""
        detections = self.detect_fire_hsv(frame)
        
        # Draw bounding boxes on frame
        annotated_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Fire {detection['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return annotated_frame, detections

def process_video_file(input_path, output_path, progress_callback=None):
    """Process video file for fire detection"""
    global processing_status
    
    try:
        processing_status['is_processing'] = True
        processing_status['progress'] = 0
        processing_status['detections'] = []
        processing_status['error'] = None
        
        detector = FireDetector()
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError("Unable to open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer with Chrome-compatible codec
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264 codec for Chrome compatibility
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        except:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec fallback
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            except:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec alternative
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                except:
                    # Last resort fallback
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, detections = detector.process_frame(frame)
            out.write(annotated_frame)
            
            # Store detections with timestamp
            timestamp = frame_count / fps
            for detection in detections:
                detection['timestamp'] = timestamp
                all_detections.append(detection)
            
            frame_count += 1
            
            # Update progress
            if progress_callback and total_frames > 0:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)
                processing_status['progress'] = progress
                processing_status['detections'] = all_detections[-10:]  # Keep last 10 detections
        
        cap.release()
        out.release()
        
        processing_status['progress'] = 100
        processing_status['detections'] = all_detections
        
        return {
            'success': True,
            'output_path': output_path,
            'total_detections': len(all_detections),
            'detections': all_detections
        }
        
    except Exception as e:
        processing_status['error'] = str(e)
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        processing_status['is_processing'] = False

@app.route('/')
def index():
    return render_template('index.html', preloaded_videos=PRELOADED_VIDEOS)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })

@app.route('/process/<filename>')
def process_video(filename):
    if processing_status['is_processing']:
        return jsonify({'error': 'Another video is currently being processed'}), 400
    
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(input_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    # Generate output filename
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Start processing in background thread
    def process_in_background():
        result = process_video_file(input_path, output_path)
        processing_status['result'] = result
        if result['success']:
            processing_status['output_filename'] = output_filename
    
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Processing started',
        'output_filename': output_filename
    })

@app.route('/process_preloaded/<filename>')
def process_preloaded(filename):
    if processing_status['is_processing']:
        return jsonify({'error': 'Another video is currently being processed'}), 400
    
    # Check if it's a preloaded video
    if not any(video['filename'] == filename for video in PRELOADED_VIDEOS):
        return jsonify({'error': 'Invalid preloaded video'}), 400
    
    input_path = filename  # Preloaded videos are in root directory
    if not os.path.exists(input_path):
        return jsonify({'error': 'Preloaded video file not found'}), 404
    
    # Generate output filename
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Start processing in background thread
    def process_in_background():
        result = process_video_file(input_path, output_path)
        processing_status['result'] = result
        if result['success']:
            processing_status['output_filename'] = output_filename
    
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Processing started',
        'output_filename': output_filename
    })

@app.route('/status')
def get_status():
    return jsonify(processing_status)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/video/<filename>')
def stream_video(filename):
    try:
        response = send_from_directory(app.config['OUTPUT_FOLDER'], filename, mimetype='video/mp4')
        # Add Chrome-specific headers
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Type'] = 'video/mp4'
        response.headers['Cache-Control'] = 'no-cache'
        return response
    except:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/preloaded/<filename>')
def stream_preloaded(filename):
    return send_from_directory('.', filename)

@app.route('/test')
def test_video():
    return send_file('test_video.html')

@app.route('/debug')
def debug_video():
    return send_file('debug_video.html')

@app.route('/chrome')
def chrome_test():
    return send_file('chrome_test.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    emit('connected', {'message': 'Connected to Fire Detection System'})

@socketio.on('disconnect')
def handle_disconnect():
    webcam_handler.stop_webcam()

@socketio.on('start_webcam')
def handle_start_webcam():
    if webcam_handler.start_webcam():
        emit('webcam_started', {'message': 'Webcam started successfully'})
    else:
        emit('webcam_error', {'error': 'Failed to start webcam'})

@socketio.on('stop_webcam')
def handle_stop_webcam():
    webcam_handler.stop_webcam()
    emit('webcam_stopped', {'message': 'Webcam stopped'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
