import cv2
import base64
import json
import threading
import time
from flask_socketio import SocketIO, emit
import numpy as np

class WebcamHandler:
    def __init__(self, socketio):
        self.socketio = socketio
        self.is_running = False
        self.current_thread = None
        self.cap = None
        
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
    
    def start_webcam(self):
        """Start webcam processing"""
        if self.is_running:
            return False
        
        self.is_running = True
        self.current_thread = threading.Thread(target=self._webcam_loop)
        self.current_thread.daemon = True
        self.current_thread.start()
        return True
    
    def stop_webcam(self):
        """Stop webcam processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.current_thread:
            self.current_thread.join(timeout=1)
    
    def _webcam_loop(self):
        """Main webcam processing loop"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.socketio.emit('webcam_error', {'error': 'Unable to access webcam'})
                return
            
            # Set webcam properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            frame_count = 0
            last_detection_time = time.time()
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, detections = self.process_frame(frame)
                
                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame data to client
                self.socketio.emit('webcam_frame', {
                    'frame': frame_data,
                    'detections': detections,
                    'timestamp': time.time()
                })
                
                # Send detection alerts
                current_time = time.time()
                if detections and (current_time - last_detection_time) > 2:  # Throttle alerts
                    self.socketio.emit('fire_alert', {
                        'detections': detections,
                        'timestamp': current_time
                    })
                    last_detection_time = current_time
                
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            self.socketio.emit('webcam_error', {'error': str(e)})
        finally:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
