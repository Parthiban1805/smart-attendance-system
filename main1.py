# --- ADDED: Suppress TensorFlow informational messages ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from datetime import timedelta
from scipy.spatial.distance import euclidean

# --- CONFIGURATION ---
VIDEO_SOURCE = r"P:\attendance\VID_20250621_154353.mp4"
OUTPUT_VIDEO_PATH = 'output_video.avi'
LINE_Y_POSITION = 800 
DETECTION_INTERVAL = 10 
KNOWN_FACES_DIR = 'dataset'
RECOGNITION_THRESHOLD = 1.0
REQUIRED_INPUT_SIZE = (160, 160)

# --- Models ---
EMBEDDER = FaceNet()
DETECTOR = MTCNN()

# --- HELPER FUNCTIONS ---
def extract_face(image, required_size=REQUIRED_INPUT_SIZE):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = DETECTOR.detect_faces(image_rgb)
    if not results:
        return None, None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None, None
    face_image = cv2.resize(face, required_size)
    return face_image, (x1, y1, width, height)

def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    samples = np.expand_dims(face_pixels, axis=0)
    embeddings = EMBEDDER.embeddings(samples)
    return embeddings[0]

def load_known_faces(directory):
    known_embeddings = {}
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            embeddings_for_person = []
            for filename in os.listdir(person_dir):
                path = os.path.join(person_dir, filename)
                try:
                    image = cv2.imread(path)
                    if image is None:
                        print(f"Warning: Could not read image {path}")
                        continue
                    face, _ = extract_face(image)
                    if face is not None:
                        embeddings_for_person.append(get_embedding(face))
                except Exception as e:
                    print(f"Error processing {path}: {e}")
            if embeddings_for_person:
                known_embeddings[person_name] = np.mean(embeddings_for_person, axis=0)
                print(f"Loaded database for: {person_name}")
    return known_embeddings

def create_tracker():
    """Create a tracker object compatible with different OpenCV versions"""
    try:
        # Try newer OpenCV versions (4.5.1+)
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
        # Try older OpenCV versions
        elif hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
        # Fallback to KCF tracker if CSRT is not available
        elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            print("Warning: CSRT tracker not available, using KCF tracker as fallback")
            return cv2.legacy.TrackerKCF_create()
        elif hasattr(cv2, 'TrackerKCF_create'):
            print("Warning: CSRT tracker not available, using KCF tracker as fallback")
            return cv2.TrackerKCF_create()
        else:
            print("Error: No compatible tracker found")
            return None
    except Exception as e:
        print(f"Error creating tracker: {e}")
        return None

# --- TRACKER CLASS FOR INDIVIDUAL TRACKERS ---
class PersonTracker:
    def __init__(self, tracker_id, name, bbox, frame):
        self.id = tracker_id
        self.name = name
        self.crossed = False
        self.last_pos = None
        self.active = True
        
        # Create individual tracker using the compatibility function
        self.tracker = create_tracker()
        
        if self.tracker is None:
            self.active = False
            print(f"Failed to create tracker for {name}")
            return
            
        success = self.tracker.init(frame, bbox)
        if not success:
            self.active = False
            print(f"Failed to initialize tracker for {name}")
    
    def update(self, frame):
        if not self.active or self.tracker is None:
            return None
        
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                return bbox
            else:
                self.active = False
                return None
        except Exception as e:
            print(f"Error updating tracker for {self.name}: {e}")
            self.active = False
            return None

# --- MAIN LOGIC ---

if __name__ == "__main__":
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test tracker creation
    test_tracker = create_tracker()
    if test_tracker is None:
        print("Error: Cannot create trackers. Please check your OpenCV installation.")
        exit()
    else:
        print("Tracker creation successful!")
    
    # 1. Load Face Database
    known_face_embeddings = load_known_faces(KNOWN_FACES_DIR)
    
    # 2. Initialize Video Capture and Writer
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_SOURCE}")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for path: {OUTPUT_VIDEO_PATH}")
        cap.release()
        exit()

    # 3. Initialize Tracking Variables
    frame_count = 0
    person_trackers = []  # List to store PersonTracker objects
    attendance_log = {}
    next_tracker_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Periodically Detect and Recognize New Faces
        if frame_count % DETECTION_INTERVAL == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_detections = DETECTOR.detect_faces(rgb_frame)

            for detection in all_detections:
                x, y, w, h = detection['box']
                if w <= 0 or h <= 0: 
                    continue
                x, y = abs(x), abs(y)
                
                face_crop_rgb = rgb_frame[y:y+h, x:x+w]
                if face_crop_rgb.size == 0:
                    continue

                face_image = cv2.resize(face_crop_rgb, REQUIRED_INPUT_SIZE)
                live_embedding = get_embedding(face_image)
                
                min_dist = float('inf')
                identity = 'Unknown'
                for name, known_emb in known_face_embeddings.items():
                    dist = euclidean(known_emb, live_embedding)
                    if dist < min_dist and dist < RECOGNITION_THRESHOLD:
                        min_dist = dist
                        identity = name

                if identity != 'Unknown' and identity not in attendance_log:
                    # Check if this person is already being tracked
                    is_already_tracked = False
                    for tracker in person_trackers:
                        if tracker.name == identity and tracker.active:
                            is_already_tracked = True
                            break
                    
                    if not is_already_tracked:
                        # Create new tracker for this person
                        bbox = (x, y, w, h)
                        new_tracker = PersonTracker(next_tracker_id, identity, bbox, frame)
                        
                        if new_tracker.active:
                            person_trackers.append(new_tracker)
                            print(f"Started tracking {identity}.")
                            next_tracker_id += 1

        # 5. Update All Active Trackers and Process Results
        active_trackers = []
        for tracker in person_trackers:
            if not tracker.active:
                continue
                
            bbox = tracker.update(frame)
            if bbox is None:
                print(f"Lost tracking for {tracker.name}")
                continue
            
            active_trackers.append(tracker)
            
            # Draw bounding box and process line crossing
            x, y, w, h = bbox
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            centroid_y = int(y + h / 2)
            
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            cv2.putText(frame, tracker.name, (p1[0], p1[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Check for line crossing
            last_y = tracker.last_pos
            
            if last_y is not None and last_y < LINE_Y_POSITION and centroid_y >= LINE_Y_POSITION:
                if not tracker.crossed:
                    timestamp = str(timedelta(seconds=frame_count/fps))
                    attendance_log[tracker.name] = timestamp
                    tracker.crossed = True
                    print(f"ATTENDANCE: {tracker.name} entered at {timestamp.split('.')[0]}")
            
            tracker.last_pos = centroid_y
        
        # Update the list to only keep active trackers
        person_trackers = active_trackers

        # 6. Draw Visualization Elements
        cv2.line(frame, (0, LINE_Y_POSITION), (frame_width, LINE_Y_POSITION), (0, 0, 255), 2)
        cv2.putText(frame, "ENTRY LINE", (10, LINE_Y_POSITION - 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        count = len(attendance_log)
        cv2.putText(frame, f"Persons Entered: {count}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video Attendance System', frame)
        out.write(frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Clean Up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n--- Final Attendance Report ---")
    if not attendance_log:
        print("No one was marked as entered.")
    else:
        for name, time in attendance_log.items():
            print(f"- {name}: Entered at {time.split('.')[0]}")
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")