import cv2
import os
import easyocr
import re
from ultralytics import YOLO

# Configuration
VIDEO_PATH = r"number_plt\sample1.mp4"  # Path to your input video
OUTPUT_DIR = r"number_plt\output"       # Directory to save output files
MODEL_PATH = r"number_plt\best.pt"                 # Path to your YOLO model

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLATE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "plate.jpg")
TEXT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "plate_text.txt")

def preprocess_image(image):
    """Preprocess the image to improve OCR accuracy."""
    if image is None:
        print("❌ Error: Invalid image for preprocessing.")
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

def detect_license_plate():
    """Detect license plates in video and perform OCR."""
    print("🔍 Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return
    
    print(f"🎬 Processing video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error: Could not open video source.")
        return
    
    plate_saved = False
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame for efficiency
            continue
            
        # Display progress
        if frame_count % 50 == 0:
            print(f"Processing frame {frame_count}...")
            
        # Detect objects in the frame
        results = model(frame)
        
        # Look for license plates (assuming class ID 3 is for license plates)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Check if it's a license plate with good confidence
                if class_id == 3 and confidence > 0.5:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Crop the license plate from the frame
                    plate_crop = frame[y1:y2, x1:x2]
                    
                    # Save the cropped license plate
                    if plate_crop.size > 0:
                        cv2.imwrite(PLATE_OUTPUT_PATH, plate_crop)
                        print(f"✅ License plate detected and saved to: {PLATE_OUTPUT_PATH}")
                        plate_saved = True
                        
                        # Perform OCR on the detected plate
                        print("🔍 Recognizing license plate text...")
                        plate_text = recognize_text(plate_crop)
                        
                        if plate_text:
                            # Save the plate text to file
                            with open(TEXT_OUTPUT_PATH, "w") as file:
                                file.write(plate_text)
                            print(f"✅ License plate text saved to: {TEXT_OUTPUT_PATH}")
                            print(f"🚘 Detected License Plate: {plate_text}")
                        else:
                            print("⚠️ Could not recognize text on the license plate.")
                        
                        break
            
            if plate_saved:
                break
                
        if plate_saved:
            break
    
    # Release video capture
    cap.release()
    cv2.destroyAllWindows()
    
    if not plate_saved:
        print("❌ No license plate detected in the video.")

if __name__ == "__main__":
    detect_license_plate()