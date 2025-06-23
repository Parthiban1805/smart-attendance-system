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
OUTPUT_VIDEO_PATH = 'output_video_final_zonal_tracking.mp4'
KNOWN_FACES_DIR = 'dataset1'
EMBEDDINGS_FILE = 'face_embeddings.pkl'
DETECTION_INTERVAL = 10
RESIZE_FACTOR = 0.5
RECOGNITION_THRESHOLD = 0.9
MAX_DISAPPEARED_FRAMES = 20
IOU_THRESHOLD = 0.4
REQUIRED_INPUT_SIZE = (160, 160)

### MODIFIED: Zonal Configuration ###
# We define two lines to create an "in-between" zone. A person must cross them
# in order to be counted. The values are ratios of the frame's height.
# Line A is the "outer" or first line. Line B is the "inner" or second line.
LINE_A_Y_RATIO = 0.45  # Outer line at 45% of frame height
LINE_B_Y_RATIO = 0.55  # Inner line at 55% of frame height

# --- Models ---
print("Initializing models...")
EMBEDDER = FaceNet()
DETECTOR = MTCNN()
print("✓ Models initialized.")

# --- HELPER FUNCTIONS ---
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

### --- MODIFIED: PersonTracker now uses a Zonal State Machine --- ###
class PersonTracker:
    def __init__(self, tracker_id, name, bbox, frame, line_a_y, line_b_y):
        self.id = tracker_id
        self.name = name
        self.line_a_y = line_a_y
        self.line_b_y = line_b_y
        
        self.disappeared_frames = 0
        self.crossing_event = None # Will be set to 'entry' or 'exit' on a valid crossing
        
        self.tracker = create_tracker()
        self.tracker.init(frame, bbox)
        self.last_bbox = bbox
        
        # Determine initial state based on starting anchor point (feet)
        x, y, w, h = bbox
        self.current_anchor_y = int(y + h)
        self.previous_anchor_y = self.current_anchor_y
        
        if self.current_anchor_y < self.line_a_y:
            self.state = "outside"
        elif self.current_anchor_y > self.line_b_y:
            self.state = "inside"
        else:
            self.state = "between"
            
        print(f"✨ New tracker ID {self.id} for {self.name}. Initial state: {self.state}")

    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            self.last_bbox = bbox
            self.disappeared_frames = 0
            
            x, y, w, h = [int(v) for v in bbox]
            self.previous_anchor_y = self.current_anchor_y
            self.current_anchor_y = int(y + h) # We only need the y-coord for this logic
            
            # If a crossing hasn't been logged yet, check for a new state change
            if not self.crossing_event:
                self._check_zonal_crossing()
        else:
            self.last_bbox = None
            self.disappeared_frames += 1
        return self.last_bbox

    ### NEW: State machine logic for zonal crossing ###
    def _check_zonal_crossing(self):
        # Check for Entry (must cross A then B, assuming top-to-bottom entry)
        # 1. Transition from Outside to Between
        if self.state == "outside" and self.previous_anchor_y < self.line_a_y and self.current_anchor_y >= self.line_a_y:
            self.state = "between"
            print(f"ID {self.id} ({self.name}) entered the 'between' zone from outside.")
        # 2. Transition from Between to Inside (This confirms an ENTRY)
        elif self.state == "between" and self.previous_anchor_y < self.line_b_y and self.current_anchor_y >= self.line_b_y:
            self.state = "inside"
            self.crossing_event = "entry"
            print(f"🚪✅ ENTRY CONFIRMED: ID {self.id} ({self.name})")

        # Check for Exit (must cross B then A)
        # 1. Transition from Inside to Between
        if self.state == "inside" and self.previous_anchor_y > self.line_b_y and self.current_anchor_y <= self.line_b_y:
            self.state = "between"
            print(f"ID {self.id} ({self.name}) left the 'inside' zone.")
        # 2. Transition from Between to Outside (This confirms an EXIT)
        elif self.state == "between" and self.previous_anchor_y > self.line_a_y and self.current_anchor_y <= self.line_a_y:
            self.state = "outside"
            self.crossing_event = "exit"
            print(f"🚪✅ EXIT CONFIRMED: ID {self.id} ({self.name})")

# --- AttendanceManager is unchanged, it just consumes the 'entry'/'exit' events ---
class AttendanceManager:
    def __init__(self):
        self.entries = {}
        self.exits = {}
        self.log_file = f"attendance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    def record_crossing(self, name, direction, timestamp):
        if direction == "entry" and name not in self.entries:
            self.entries[name] = timestamp.isoformat()
            print(f"LOG: Entry recorded for {name}.")
            self.save_log()
        elif direction == "exit" and name not in self.exits:
            self.exits[name] = timestamp.isoformat()
            print(f"LOG: Exit recorded for {name}.")
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
    print("\n🎯 ADVANCED ATTENDANCE SYSTEM (TWO-LINE ZONAL LOGIC)")
    known_faces = load_known_faces(KNOWN_FACES_DIR, EMBEDDINGS_FILE)
    if not known_faces: return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"✗ Error opening video: {VIDEO_SOURCE}")
        return

    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25

    ### MODIFIED: Set up the two lines based on frame height and configured ratios ###
    line_a_y = int(h * LINE_A_Y_RATIO)
    line_b_y = int(h * LINE_B_Y_RATIO)
    print(f"✓ Using zonal lines. Outer Line (A) at y={line_a_y}, Inner Line (B) at y={line_b_y}")

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

            ### NEW: Draw both lines for visualization ###
            cv2.line(frame, (0, line_a_y), (w, line_a_y), (0, 255, 255), 2)
            cv2.putText(frame, "LINE A (Outer)", (10, line_a_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.line(frame, (0, line_b_y), (w, line_b_y), (255, 255, 0), 2)
            cv2.putText(frame, "LINE B (Inner)", (10, line_b_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Update all trackers and check if they have generated a crossing event
            for tracker in trackers:
                tracker.update(frame)
                if tracker.crossing_event:
                    manager.record_crossing(tracker.name, tracker.crossing_event, current_time)
                    tracker.crossing_event = None # Reset after logging to prevent re-logging

            # Face detection and tracker association logic
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

                ### MODIFIED: Create new trackers using the new Zonal constructor ###
                for i in unmatched_detections_indices:
                    det = detections[i]
                    is_already_tracked = any(t.name == det['name'] and t.disappeared_frames < 5 for t in trackers)
                    if not is_already_tracked:
                        new_tracker = PersonTracker(next_id, det['name'], det['bbox'], frame, line_a_y, line_b_y)
                        trackers.append(new_tracker)
                        next_id += 1

            # Update list of active trackers and draw their info on the frame
            active_trackers = []
            for tracker in trackers:
                if tracker.disappeared_frames < MAX_DISAPPEARED_FRAMES:
                    active_trackers.append(tracker)
                    if tracker.last_bbox:
                        x, y, w_box, h_box = [int(v) for v in tracker.last_bbox]
                        
                        # Color the bounding box based on the person's current state
                        if tracker.state == "outside": color = (0, 0, 255)   # Red
                        elif tracker.state == "between": color = (0, 255, 255) # Yellow
                        else: color = (0, 255, 0) # Green

                        cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)
                        label = f"ID {tracker.id}: {tracker.name} ({tracker.state})"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    print(f"✗ Tracker ID {tracker.id} ({tracker.name}) removed (disappeared).")
            trackers = active_trackers

            # Display stats and write frame
            stats = manager.get_stats()
            info_text = f"Entries: {stats['total_entries']} | Exits: {stats['total_exits']} | Occupancy: {stats['occupancy']}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            out.write(frame)
            cv2.imshow('Attendance System (Zonal Tracking)', frame)
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