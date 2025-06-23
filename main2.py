# --- ADDED: Suppress TensorFlow informational messages ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from datetime import timedelta, datetime
from scipy.spatial.distance import euclidean
import json
import math
import sys
from pathlib import Path

# --- CONFIGURATION ---
VIDEO_SOURCE = r"D:\attendance\Person monitoring video\VID_20250621_154440.mp4"
OUTPUT_VIDEO_PATH = 'output_video_final_personal_tripwire.mp4' # Changed output name
KNOWN_FACES_DIR = 'dataset1'
EMBEDDINGS_FILE = 'face_embeddings.pkl'
DETECTION_INTERVAL = 10
RESIZE_FACTOR = 0.5
RECOGNITION_THRESHOLD = 0.9
MAX_DISAPPEARED_FRAMES = 20
IOU_THRESHOLD = 0.4
REQUIRED_INPUT_SIZE = (160, 160)

### NEW: Configuration for the new per-person tripwire logic ###
# A person must move this percentage of the frame's height from their starting
# point to trigger a crossing event. Prevents false triggers from small movements.
VERTICAL_CROSSING_THRESHOLD_RATIO = 0.15
# Set to True if "entry" means moving from the top of the frame towards the bottom.
# Set to False if "entry" means moving from the bottom of the frame towards the top.
ENTRY_IS_DOWN = True


# --- Models ---
print("Initializing models...")
EMBEDDER = FaceNet()
DETECTOR = MTCNN()
print("✓ Models initialized.")

# --- HELPER FUNCTIONS --- (No changes here, keeping for completeness)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator != 0 else 0.0

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
    return EMBEDDER.embeddings(samples)[0]

def get_dataset_modification_time(directory):
    latest_time = 0
    if not os.path.exists(directory): return latest_time
    for root, _, files in os.walk(directory):
        for file in files:
            try: latest_time = max(latest_time, os.path.getmtime(os.path.join(root, file)))
            except OSError: continue
    return latest_time

def save_embeddings(embeddings, filepath):
    try:
        with open(filepath, 'wb') as f: pickle.dump(embeddings, f)
        print(f"✓ Embeddings saved to {filepath}")
    except Exception as e: print(f"✗ Error saving embeddings: {e}")

def load_embeddings(filepath):
    try:
        with open(filepath, 'rb') as f: return pickle.load(f)
    except Exception as e:
        print(f"✗ Error loading embeddings: {e}")
        return None

def should_recompute_embeddings(dataset_dir, embeddings_file):
    if not os.path.exists(embeddings_file) or get_dataset_modification_time(dataset_dir) > os.path.getmtime(embeddings_file):
        print("Dataset changed or embeddings not found. Recomputing...")
        return True
    print("Using cached embeddings.")
    return False

def load_known_faces(directory, embeddings_file):
    if should_recompute_embeddings(directory, embeddings_file):
        known_embeddings = compute_embeddings(directory)
        if known_embeddings: save_embeddings(known_embeddings, embeddings_file)
        return known_embeddings
    return load_embeddings(embeddings_file)

def compute_embeddings(directory):
    known_embeddings = {}
    if not os.path.exists(directory): return known_embeddings
    print(f"Processing dataset directory: {directory}")
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            embeddings_for_person = []
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(person_dir, filename)
                    image = cv2.imread(path)
                    if image is not None:
                        face, _ = extract_face(image)
                        if face is not None: embeddings_for_person.append(get_embedding(face))
            if embeddings_for_person:
                known_embeddings[person_name] = np.mean(embeddings_for_person, axis=0)
    return known_embeddings

def create_tracker():
    try:
        if hasattr(cv2, 'legacy'): return cv2.legacy.TrackerCSRT_create()
        return cv2.TrackerCSRT_create()
    except: return None

### MODIFIED: LineDetector class has been removed as it's no longer needed. ###

### --- MODIFIED: PersonTracker now uses a per-person vertical tripwire --- ###
class PersonTracker:
    def __init__(self, tracker_id, name, bbox, frame, vertical_threshold, entry_is_down):
        self.id = tracker_id
        self.name = name
        self.vertical_threshold = vertical_threshold
        self.entry_is_down = entry_is_down
        
        self.disappeared_frames = 0
        self.has_crossed_line = False
        self.crossing_direction = None
        
        self.tracker = create_tracker()
        self.tracker.init(frame, bbox)
        self.last_bbox = bbox
        
        # Set the initial position and the personal "tripwire"
        x, y, w, h = bbox
        initial_anchor = (int(x + w/2), int(y + h))
        self.initial_anchor_y = initial_anchor[1]
        self.current_anchor = initial_anchor
        self.tripwire_y = self.initial_anchor_y # For visualization

    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            self.last_bbox = bbox
            self.disappeared_frames = 0
            
            x, y, w, h = [int(v) for v in bbox]
            self.current_anchor = (int(x + w/2), int(y + h))
            
            if not self.has_crossed_line:
                self._check_vertical_crossing()
        else:
            self.last_bbox = None
            self.disappeared_frames += 1
        return self.last_bbox

    ### NEW: Replaces the old line-crossing logic with a dynamic, per-person approach ###
    def _check_vertical_crossing(self):
        current_y = self.current_anchor[1]
        vertical_distance_moved = current_y - self.initial_anchor_y

        # Check if the person has moved a significant vertical distance
        if abs(vertical_distance_moved) > self.vertical_threshold:
            self.has_crossed_line = True
            
            is_moving_down = vertical_distance_moved > 0
            
            # Determine direction based on configuration
            if (is_moving_down and self.entry_is_down) or \
               (not is_moving_down and not self.entry_is_down):
                self.crossing_direction = "entry"
            else:
                self.crossing_direction = "exit"
                
            print(f"🚶✅ ID {self.id} ({self.name}) crossed their vertical threshold! Direction: {self.crossing_direction}")

# --- AttendanceManager and other helpers remain the same ---
class AttendanceManager:
    def __init__(self):
        self.entries = {} # Use a dict to store entry times
        self.exits = {}   # Use a dict to store exit times
        self.log_file = f"attendance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    def record_crossing(self, name, direction, timestamp):
        if direction == "entry" and name not in self.entries:
            self.entries[name] = timestamp.isoformat()
            print(f"🚪 ENTRY Recorded: {name} at {timestamp.strftime('%H:%M:%S')}")
            self.save_log()
        elif direction == "exit" and name not in self.exits:
            self.exits[name] = timestamp.isoformat()
            print(f"🚪 EXIT Recorded: {name} at {timestamp.strftime('%H:%M:%S')}")
            self.save_log()

    def save_log(self):
        log_data = {'entries': self.entries, 'exits': self.exits}
        with open(self.log_file, 'w') as f: json.dump(log_data, f, indent=4)

    def get_stats(self):
        entered_set = set(self.entries.keys())
        exited_set = set(self.exits.keys())
        currently_inside = entered_set - exited_set
        return {
            'total_entries': len(self.entries),
            'total_exits': len(self.exits),
            'occupancy': len(currently_inside)
        }
    def print_final_report(self):
        print("\n" + "="*60 + "\nFINAL REPORT\n" + "="*60)
        stats = self.get_stats()
        print(f"Total Unique Entries: {stats['total_entries']}")
        for name, time in self.entries.items(): print(f"  - {name} at {time}")
        print(f"\nTotal Unique Exits: {stats['total_exits']}")
        for name, time in self.exits.items(): print(f"  - {name} at {time}")

        print(f"\nFinal Occupancy: {stats['occupancy']}")
        print(f"Log file: {self.log_file}\n" + "="*60)


def main():
    print("\n🎯 ADVANCED ATTENDANCE SYSTEM (PERSONAL TRIPWIRE LOGIC)")
    known_faces = load_known_faces(KNOWN_FACES_DIR, EMBEDDINGS_FILE)
    if not known_faces: return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"✗ Error opening video: {VIDEO_SOURCE}")
        return

    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25 # Default fps if not available
    
    ### MODIFIED: Calculate the absolute pixel threshold from the ratio ###
    vertical_threshold_pixels = int(h * VERTICAL_CROSSING_THRESHOLD_RATIO)
    print(f"✓ Using a vertical crossing threshold of {vertical_threshold_pixels} pixels.")
    
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count, next_id = 0, 0
    trackers, manager = [], AttendanceManager()
    start_time = datetime.now()

    print("\n🔄 PROCESSING VIDEO...\n" + "-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            current_time = start_time + timedelta(seconds=frame_count / fps)

            ### MODIFIED: Update trackers and record crossings based on the new logic ###
            for tracker in trackers:
                tracker.update(frame)
                if tracker.has_crossed_line:
                    # Record the crossing only once
                    manager.record_crossing(tracker.name, tracker.crossing_direction, current_time)
                    # To prevent re-recording, you could add a flag, but the current manager logic already handles this.

            if frame_count % DETECTION_INTERVAL == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                detections = []
                for face_info in DETECTOR.detect_faces(rgb_frame):
                    x, y, w_face, h_face = [int(v / RESIZE_FACTOR) for v in face_info['box']]
                    face_crop = frame[y:y+h_face, x:x+w_face]
                    if face_crop.size == 0: continue
                    face_img, _ = extract_face(face_crop)
                    if face_img is None: continue
                    
                    emb = get_embedding(face_img)
                    min_dist, identity = float('inf'), 'Unknown'
                    for name, known_emb in known_faces.items():
                        dist = euclidean(emb, known_emb)
                        if dist < RECOGNITION_THRESHOLD and dist < min_dist:
                            min_dist, identity = dist, name
                    
                    if identity != 'Unknown':
                        detections.append({'name': identity, 'bbox': (x, y, w_face, h_face)})
            
                unmatched_detections_indices = list(range(len(detections)))
                if detections:
                    matched_tracker_indices = set()
                    for i, det in enumerate(detections):
                        best_iou, best_match_idx = 0, -1
                        for j, tracker in enumerate(trackers):
                            if tracker.last_bbox is None: continue
                            iou = calculate_iou(det['bbox'], tracker.last_bbox)
                            if iou > IOU_THRESHOLD and iou > best_iou:
                                best_iou, best_match_idx = iou, j
                        
                        if best_match_idx != -1 and best_match_idx not in matched_tracker_indices:
                            trackers[best_match_idx].tracker.init(frame, det['bbox'])
                            trackers[best_match_idx].last_bbox = det['bbox']
                            trackers[best_match_idx].disappeared_frames = 0
                            if i in unmatched_detections_indices: unmatched_detections_indices.remove(i)
                            matched_tracker_indices.add(best_match_idx)

                ### MODIFIED: Create new trackers using the new constructor ###
                for i in unmatched_detections_indices:
                    det = detections[i]
                    is_already_tracked = any(t.name == det['name'] and t.disappeared_frames < 5 for t in trackers)
                    if not is_already_tracked:
                        # Pass the threshold and direction flag to the new tracker
                        new_tracker = PersonTracker(next_id, det['name'], det['bbox'], frame, vertical_threshold_pixels, ENTRY_IS_DOWN)
                        trackers.append(new_tracker)
                        print(f"✨ New tracker created: ID {next_id} for {det['name']} with tripwire at y={new_tracker.tripwire_y}")
                        next_id += 1

            active_trackers = []
            for tracker in trackers:
                if tracker.disappeared_frames < MAX_DISAPPEARED_FRAMES:
                    active_trackers.append(tracker)
                    if tracker.last_bbox:
                        color = (255, 255, 0) # Default tracking color
                        if tracker.has_crossed_line:
                            color = (0, 255, 0) if tracker.crossing_direction == "entry" else (0, 0, 255)
                        
                        x, y, w_box, h_box = [int(v) for v in tracker.last_bbox]
                        cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)
                        cv2.putText(frame, f"ID {tracker.id}: {tracker.name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Visualize the anchor point (feet)
                        cv2.circle(frame, tracker.current_anchor, 5, color, -1)

                        ### NEW: Visualize the personal tripwire for each person ###
                        if not tracker.has_crossed_line:
                             # Draw a dashed line for the personal tripwire
                            for i in range(x, x + w_box, 15):
                                cv2.line(frame, (i, tracker.tripwire_y), (i + 5, tracker.tripwire_y), (255, 0, 255), 2)
                            
                else:
                    print(f"✗ Tracker ID {tracker.id} ({tracker.name}) removed (disappeared).")
            trackers = active_trackers

            stats = manager.get_stats()
            info_text = f"Entries: {stats['total_entries']} | Exits: {stats['total_exits']} | Occupancy: {stats['occupancy']}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            out.write(frame)
            cv2.imshow('Attendance System (Personal Tripwire)', frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        print("\n✓ Processing finished.")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    manager.print_final_report()

if __name__ == '__main__':
    main()