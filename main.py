import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine
from datetime import datetime
import csv

# --- INITIALIZATION ---

# Initialize FaceNet for embeddings and MTCNN for face detection
print("Initializing models...")
embedder = FaceNet()
detector = MTCNN()

# --- STATE MANAGEMENT & LOGGING ---
# This set will hold the names of people who have entered.
persons_in_room = set()
attendance_file = 'attendance_log.csv'

# --- DATABASE PREPARATION ---

def get_face_embedding(image):
    """Detects a face, extracts it, and returns its embedding."""
    faces = detector.detect_faces(image)
    if not faces:
        return None

    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1 + width, y1 + height
    face_pixels = image[y1:y2, x1:x2]
    
    face_image = cv2.resize(face_pixels, (160, 160))
    face_image = face_image.astype('float32')
    
    mean, std = face_image.mean(), face_image.std()
    face_image = (face_image - mean) / std
    
    face_image = np.expand_dims(face_image, axis=0)
    embedding = embedder.embeddings(face_image)
    return embedding[0]

def load_known_faces(database_path='dataset'):
    """Loads faces from the database and calculates their embeddings."""
    known_embeddings = []
    known_names = []
    
    print("Loading known faces from database...")
    for person_name in os.listdir(database_path):
        person_dir = os.path.join(database_path, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                filepath = os.path.join(person_dir, filename)
                image = cv2.imread(filepath)
                if image is None: continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                embedding = get_face_embedding(image_rgb)
                
                if embedding is not None:
                    known_embeddings.append(embedding)
                    known_names.append(person_name)
                    print(f"- Loaded embedding for {person_name} from {filename}")
    
    return np.array(known_embeddings), known_names

# Load the database on startup
KNOWN_EMBEDDINGS, KNOWN_NAMES = load_known_faces()
print("\nDatabase loading complete!")

# --- ATTENDANCE LOGIC ---

# Create CSV file with header if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Name', 'Action', 'PeopleInRoomCount'])

def log_entry(person_name):
    """
    Checks if the person is already in the room. If not, logs their entry
    and adds them to the state.
    """
    global persons_in_room

    if person_name not in persons_in_room:
        persons_in_room.add(person_name)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_count = len(persons_in_room)
        
        with open(attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, person_name, "ENTERED", current_count])
            
        print(f"LOG: {timestamp} - {person_name} ENTERED. People in room: {current_count}")

# --- FRAME PROCESSING ---

def process_frame(frame):
    """Processes a single frame for face recognition and triggers entry logging."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)
    
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        
        face_pixels = frame_rgb[y1:y2, x1:x2]
        live_embedding = get_face_embedding(face_pixels)
        if live_embedding is None: continue
            
        min_dist = float('inf')
        identity = 'Unknown'
        
        for i, known_embedding in enumerate(KNOWN_EMBEDDINGS):
            dist = cosine(known_embedding, live_embedding)
            if dist < min_dist:
                min_dist = dist
                identity = KNOWN_NAMES[i]
        
        # Threshold for recognition
        recognition_threshold = 0.5 
        
        person_name = "Unknown"
        if min_dist <= recognition_threshold:
            person_name = identity
            # This is the core logic: try to log the entry.
            log_entry(person_name)

        # Draw bounding box and name on the frame
        color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{person_name} ({min_dist:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

# --- MAIN LOOP ---

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Replace with the path to your entry video file.
    # You can also use a camera index like 0 for a live webcam.
    entry_video_path = "entry_input.mp4" 
    
    cap = cv2.VideoCapture(entry_video_path) 

    if not cap.isOpened():
        print(f"Error: Could not open video file: {entry_video_path}")
        exit()

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If the video has ended, break the loop
        if not ret:
            print("Video file has finished.")
            break

        # Process the frame
        processed_frame = process_frame(frame)
        
        # Display info on the video window
        cv2.putText(processed_frame, "ENTRY TEST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        room_info = f"In Room ({len(persons_in_room)}): {', '.join(persons_in_room)}"
        cv2.putText(processed_frame, room_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Entry Test Video', processed_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete. Attendance log saved to attendance_log.csv")