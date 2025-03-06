from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import os
import easyocr
import re
from ultralytics import YOLO
import uuid
import time
import threading
import numpy as np

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = "number_plt/best.pt"  # Path to your YOLO model
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Processing status dictionary to track ongoing jobs
processing_jobs = {}

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess the image to improve OCR accuracy."""
    if image is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Apply mild blur to reduce noise
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    
    return blur

def recognize_text(image):
    """Recognize text from a license plate image."""
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a GPU
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    if processed_image is None:
        return ""
    
    # Perform OCR on both original and processed images
    results_processed = reader.readtext(processed_image, detail=1, 
                                      allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    results_original = reader.readtext(image, detail=1, 
                                     allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    
    # Combine results and filter by confidence
    all_results = results_processed + results_original
    filtered_results = [(bbox, text, conf) for bbox, text, conf in all_results if conf > 0.5]
    
    # Sort results from left to right
    filtered_results.sort(key=lambda item: item[0][0][0])
    
    # Combine text parts
    if filtered_results:
        plate_text = " ".join([text for (_, text, _) in filtered_results])
        
        # Clean up the text
        plate_text = re.sub(r'\s+', '', plate_text)  # Remove all whitespace
        plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)  # Keep only alphanumeric
        
        return plate_text
    
    return ""

def process_video(job_id, video_path):
    """Process video to detect license plates, helmets, and riders"""
    try:
        # Load YOLO model
        model = YOLO(MODEL_PATH)
        
        # Create job output directory
        job_output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        os.makedirs(job_output_dir, exist_ok=True)
        
        # Update job status
        processing_jobs[job_id]['status'] = 'processing'
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['message'] = 'Could not open video file'
            return
        
        frame_count = 0
        plate_detected = False
        helmet_detected = False
        rider_detected = False
        frame_image_path = None
        
        # Initialize results data with default values
        results_data = {
            'plate_text': 'No text recognized',
            'plate_image': None,
            'helmet_detected': False,
            'rider_detected': False,
            'frame_image': None,
            'message': 'Processing...'
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 5 != 0:  # Process every 5th frame for efficiency
                continue
                
            # Detect objects in the frame
            results = model(frame)
            
            # Process detection results
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # License plate (assuming class ID 3 is for license plates)
                    if class_id == 3 and confidence > 0.5 and not plate_detected:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        plate_crop = frame[y1:y2, x1:x2]
                        
                        if plate_crop.size > 0:
                            plate_path = os.path.join(job_output_dir, "plate.jpg")
                            cv2.imwrite(plate_path, plate_crop)
                            plate_detected = True
                            results_data['plate_image'] = f"/outputs/{job_id}/plate.jpg"
                            
                            # Perform OCR
                            plate_text = recognize_text(plate_crop)
                            if plate_text:
                                results_data['plate_text'] = plate_text
                            
                    # Helmet detection (assuming class ID 1 is for helmets)
                    elif class_id == 1 and confidence > 0.5:
                        helmet_detected = True
                        results_data['helmet_detected'] = True
                    
                    # Rider detection (assuming class ID 0 is for riders/persons)
                    elif class_id == 0 and confidence > 0.5:
                        rider_detected = True
                        results_data['rider_detected'] = True
            
            # Save a frame with annotations
            if frame_count % 30 == 0:
                annotated_frame = results[0].plot()
                frame_filename = f"annotated_{frame_count}.jpg"
                frame_path = os.path.join(job_output_dir, frame_filename)
                cv2.imwrite(frame_path, annotated_frame)
                
                # Save the latest frame for display
                frame_image_path = f"/outputs/{job_id}/{frame_filename}"
        
        # Release video capture
        cap.release()
        
        # Add the last saved frame to results
        if frame_image_path:
            results_data['frame_image'] = frame_image_path
        
        # Update final results
        results_data['message'] = 'Processing completed'
        if not plate_detected:
            results_data['message'] = 'No license plate detected'
        
        # Explicitly set detection flags in results_data
        results_data['helmet_detected'] = helmet_detected
        results_data['rider_detected'] = rider_detected
        
        # Save the results to a JSON file for persistence
        processing_jobs[job_id]['results'] = results_data
        processing_jobs[job_id]['status'] = 'completed'
        
    except Exception as e:
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['message'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<job_id>/<filename>')
def output_file(job_id, filename):
    return send_from_directory(os.path.join(OUTPUT_FOLDER, job_id), filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Save the uploaded file
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{job_id}.{ext}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Initialize job status
        processing_jobs[job_id] = {
            'status': 'queued',
            'file_path': file_path,
            'timestamp': time.time(),
            'results': None,
            'message': 'Job queued for processing'
        }
        
        # Start processing in a background thread
        thread = threading.Thread(target=process_video, args=(job_id, file_path))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Video uploaded and queued for processing'
        })
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job_info = processing_jobs[job_id]
    response = {
        'job_id': job_id,
        'status': job_info['status'],
        'message': job_info.get('message', '')
    }
    
    # Include results if processing is completed
    if job_info['status'] == 'completed' and job_info.get('results'):
        response['results'] = job_info['results']
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)