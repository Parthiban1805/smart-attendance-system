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
import multiprocessing
import time

# --- CONFIGURATION ---
VIDEO_SOURCE_ENTRY = r"P:\attendance\Person monitoring video\VID_20250621_154440.mp4"
# Make sure this path is correct for your system
VIDEO_SOURCE_EXIT = r"P:\attendance\Person monitoring video\VID_20250621_154418.mp4" # Using your other video for testing
OUTPUT_VIDEO_ENTRY = 'output_video_entry_cam.mp4'
OUTPUT_VIDEO_EXIT = 'output_video_exit_cam.mp4'
SHARED_LOG_FILE = f"attendance_log_shared_{datetime.now().strftime('%Y%m%d')}.json"

KNOWN_FACES_DIR = 'dataset'
EMBEDDINGS_FILE = 'face_embeddings.pkl'
DETECTION_INTERVAL = 10
RESIZE_FACTOR = 0.5
RECOGNITION_THRESHOLD = 0.9
MAX_DISAPPEARED_FRAMES = 20
IOU_THRESHOLD = 0.4
REQUIRED_INPUT_SIZE = (160, 160)
VERTICAL_CROSSING_THRESHOLD_RATIO = 0.15

# --- Models (will be initialized PER PROCESS) ---
EMBEDDER = None
DETECTOR = None

# --- HELPER FUNCTIONS ---
def calculate_iou(boxA, boxB):
    # (Code is unchanged, keeping it short for brevity)
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]); yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator != 0 else 0.0

# These helpers now rely on the global DETECTOR/EMBEDDER within their process
def extract_face(image, required_size=REQUIRED_INPUT_SIZE):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = DETECTOR.detect_faces(image_rgb)
    if not results: return None, None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1); x2, y2 = x1 + width, y1 + height
    face = image_rgb[y1:y2, x1:x2]
    return (cv2.resize(face, required_size), (x1,y1,width,height)) if face.size != 0 else (None, None)

def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    samples = np.expand_dims(face_pixels, axis=0)
    return EMBEDDER.embeddings(samples)[0]

### MODIFIED: This function now accepts a lock to prevent race conditions ###
def load_known_faces(directory, embeddings_file, lock):
    def get_dataset_modification_time(directory):
        latest_time = 0
        if not os.path.exists(directory): return latest_time
        for root, _, files in os.walk(directory):
            for file in files:
                try: latest_time = max(latest_time, os.path.getmtime(os.path.join(root, file)))
                except OSError: continue
        return latest_time

    def compute_embeddings(directory):
        known_embeddings = {}
        if not os.path.exists(directory): return known_embeddings
        print(f"[{multiprocessing.current_process().name}] Processing dataset directory: {directory}")
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
                if embeddings_for_person: known_embeddings[person_name] = np.mean(embeddings_for_person, axis=0)
        return known_embeddings

    # Check if we need to recompute
    embeddings_exist = os.path.exists(embeddings_file)
    recompute = not embeddings_exist or get_dataset_modification_time(directory) > os.path.getmtime(embeddings_file)

    if recompute:
        print(f"[{multiprocessing.current_process().name}] Dataset changed or embeddings not found. Attempting to recompute...")
        # Acquire lock to ensure only one process computes
        with lock:
            # Double-check after acquiring the lock, in case another process just finished
            embeddings_exist = os.path.exists(embeddings_file)
            still_needs_recompute = not embeddings_exist or get_dataset_modification_time(directory) > os.path.getmtime(embeddings_file)
            if still_needs_recompute:
                known_embeddings = compute_embeddings(directory)
                if known_embeddings:
                    with open(embeddings_file, 'wb') as f: pickle.dump(known_embeddings, f)
                    print(f"[{multiprocessing.current_process().name}] ✓ Embeddings saved to {embeddings_file}")
                return known_embeddings
            else:
                 print(f"[{multiprocessing.current_process().name}] Another process finished re-computing. Loading new file.")

    # If we get here, the file exists and is up-to-date
    print(f"[{multiprocessing.current_process().name}] Using cached embeddings.")
    with open(embeddings_file, 'rb') as f: return pickle.load(f)

def create_tracker():
    try:
        return cv2.legacy.TrackerCSRT_create() if hasattr(cv2, 'legacy') else cv2.TrackerCSRT_create()
    except: return None

# --- SHARED STATE MANAGEMENT (Unchanged) ---
class AttendanceManager:
    # (Code is unchanged)
    def __init__(self, log_file, lock, shared_entries_dict, shared_exits_dict):
        self.log_file = log_file; self.lock = lock
        self.entries = shared_entries_dict; self.exits = shared_exits_dict
        self.load_log()
    def load_log(self):
        with self.lock:
            if os.path.exists(self.log_file):
                try:
                    with open(self.log_file, 'r') as f:
                        log_data = json.load(f)
                        self.entries.update(log_data.get('entries', {})); self.exits.update(log_data.get('exits', {}))
                    print(f"✓ Attendance log loaded from {self.log_file}")
                except (json.JSONDecodeError, IOError) as e: print(f"✗ Could not load or parse log file: {e}. Starting fresh.")
    def record_crossing(self, name, direction, timestamp, camera_id="CAM"):
        is_new_record = False
        if direction == "entry" and self.is_outside(name):
            self.entries[name] = timestamp.isoformat()
            if name in self.exits: del self.exits[name]
            is_new_record = True
            print(f"[{camera_id}] 🚪 ENTRY Recorded: {name} at {timestamp.strftime('%H:%M:%S')}")
        elif direction == "exit" and self.is_inside(name):
            self.exits[name] = timestamp.isoformat()
            is_new_record = True
            print(f"[{camera_id}] 🚪 EXIT Recorded: {name} at {timestamp.strftime('%H:%M:%S')}")
        if is_new_record: self.save_log()
    def is_inside(self, name): return name in self.entries and name not in self.exits
    def is_outside(self, name): return not self.is_inside(name)
    def save_log(self):
        with self.lock:
            log_data = {'entries': dict(self.entries), 'exits': dict(self.exits)}
            with open(self.log_file, 'w') as f: json.dump(log_data, f, indent=4)
    def get_stats(self):
        entered_set = set(self.entries.keys()); exited_set = set(self.exits.keys())
        return {'total_entries': len(self.entries), 'total_exits': len(self.exits), 'occupancy': len(entered_set - exited_set)}
    def print_final_report(self):
        print("\n" + "="*60 + "\nFINAL REPORT\n" + "="*60)
        stats = self.get_stats()
        print(f"Total Unique Entries Recorded: {stats['total_entries']}")
        for name, time in sorted(self.entries.items(), key=lambda item: item[1]): print(f"  - {name} at {time}")
        print(f"\nTotal Unique Exits Recorded: {stats['total_exits']}")
        for name, time in sorted(self.exits.items(), key=lambda item: item[1]): print(f"  - {name} at {time}")
        still_inside = list(set(self.entries.keys()) - set(self.exits.keys()))
        print(f"\nFinal Occupancy: {len(still_inside)}")
        if still_inside: print("People still inside:", ", ".join(still_inside))
        print(f"\nLog file: {self.log_file}\n" + "="*60)

# --- PERSON TRACKER (Unchanged) ---
class PersonTracker:
    # (Code is unchanged)
    def __init__(self, tracker_id, name, bbox, frame, vertical_threshold):
        self.id = tracker_id; self.name = name; self.vertical_threshold = vertical_threshold
        self.disappeared_frames = 0; self.has_crossed_line = False
        self.tracker = create_tracker(); self.tracker.init(frame, bbox)
        self.last_bbox = bbox
        x, y, w, h = bbox
        self.initial_anchor_y = int(y + h)
        self.current_anchor = (int(x + w/2), int(y + h))
        self.tripwire_y = self.initial_anchor_y
    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            self.last_bbox = bbox; self.disappeared_frames = 0
            x, y, w, h = [int(v) for v in bbox]
            self.current_anchor = (int(x + w/2), int(y + h))
            if not self.has_crossed_line: self._check_vertical_crossing()
        else:
            self.last_bbox = None; self.disappeared_frames += 1
        return self.last_bbox
    def _check_vertical_crossing(self):
        if abs(self.current_anchor[1] - self.initial_anchor_y) > self.vertical_threshold:
            self.has_crossed_line = True
            print(f"🚶✅ ID {self.id} ({self.name}) crossed their vertical threshold!")

### REFACTORED: Core processing logic, now fully self-contained ###
def process_video_stream(video_source, output_video_path, direction, shared_manager, data_lock):
    # Make models global WITHIN THIS PROCESS so helper functions can use them
    global DETECTOR, EMBEDDER
    
    # 1. Initialize models IN THIS PROCESS
    print(f"[{direction.upper()} CAM] Initializing models...")
    DETECTOR = MTCNN()
    EMBEDDER = FaceNet()
    print(f"[{direction.upper()} CAM] ✓ Models initialized.")
    
    # 2. Load known faces IN THIS PROCESS (using the lock)
    known_faces = load_known_faces(KNOWN_FACES_DIR, EMBEDDINGS_FILE, data_lock)
    if not known_faces:
        print(f"[{direction.upper()} CAM] ✗ No known faces loaded. Exiting process.")
        return

    camera_id = f"{direction.upper()} CAM"
    print(f"[{camera_id}] Starting processing for {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[{camera_id}] ✗ Error opening video: {video_source}")
        return

    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25
    
    vertical_threshold_pixels = int(h * VERTICAL_CROSSING_THRESHOLD_RATIO)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count, next_id, trackers = 0, 0, []
    start_time = datetime.now()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            current_time = start_time + timedelta(seconds=frame_count / fps)
            
            for tracker in list(trackers):
                tracker.update(frame)
                if tracker.has_crossed_line:
                    shared_manager.record_crossing(tracker.name, direction, current_time, camera_id)
                    trackers.remove(tracker)
                    print(f"[{camera_id}] ✓ Tracker ID {tracker.id} ({tracker.name}) processed and removed.")

            if frame_count % DETECTION_INTERVAL == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
                # Face detection and recognition logic... (this part is unchanged)
                detections = []
                face_crops_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = DETECTOR.detect_faces(face_crops_rgb)
                for face_info in results:
                    x, y, w_face, h_face = [int(v / RESIZE_FACTOR) for v in face_info['box']]
                    face_crop_bgr = frame[y:y+h_face, x:x+w_face]
                    if face_crop_bgr.size == 0: continue
                    face_img, _ = extract_face(face_crop_bgr) # extract_face needs BGR
                    if face_img is None: continue
                    
                    emb = get_embedding(face_img)
                    min_dist, identity = float('inf'), 'Unknown'
                    for name, known_emb in known_faces.items():
                        dist = euclidean(emb, known_emb)
                        if dist < RECOGNITION_THRESHOLD and dist < min_dist:
                            min_dist, identity = dist, name
                    
                    if identity != 'Unknown':
                        detections.append({'name': identity, 'bbox': (x, y, w_face, h_face)})
            
                # Tracker matching logic (this part is unchanged)
                matched_tracker_indices = set()
                unmatched_detections_indices = list(range(len(detections)))
                if detections:
                    for i, det in enumerate(detections):
                        best_iou, best_match_idx = 0, -1
                        for j, tracker in enumerate(trackers):
                            if tracker.last_bbox is None: continue
                            iou = calculate_iou(det['bbox'], tracker.last_bbox)
                            if iou > IOU_THRESHOLD and iou > best_iou and j not in matched_tracker_indices:
                                best_iou, best_match_idx = iou, j
                        if best_match_idx != -1:
                            trackers[best_match_idx].tracker.init(frame, det['bbox'])
                            trackers[best_match_idx].last_bbox = det['bbox']
                            trackers[best_match_idx].disappeared_frames = 0
                            if i in unmatched_detections_indices: unmatched_detections_indices.remove(i)
                            matched_tracker_indices.add(best_match_idx)

                for i in unmatched_detections_indices:
                    det = detections[i]
                    is_already_tracked = any(t.name == det['name'] for t in trackers)
                    if direction == 'entry' and shared_manager.is_inside(det['name']): continue
                    if direction == 'exit' and shared_manager.is_outside(det['name']): continue
                    if not is_already_tracked:
                        new_tracker = PersonTracker(next_id, det['name'], det['bbox'], frame, vertical_threshold_pixels)
                        trackers.append(new_tracker)
                        print(f"[{camera_id}] ✨ New tracker: ID {next_id} for {det['name']} with tripwire at y={new_tracker.tripwire_y}")
                        next_id += 1

            # Drawing and cleanup logic (this part is unchanged)
            active_trackers = []
            for tracker in trackers:
                if tracker.disappeared_frames < MAX_DISAPPEARED_FRAMES:
                    active_trackers.append(tracker)
                    if tracker.last_bbox:
                        x, y, w_box, h_box = [int(v) for v in tracker.last_bbox]
                        color = (255, 255, 0); cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)
                        cv2.putText(frame, f"ID {tracker.id}: {tracker.name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.circle(frame, tracker.current_anchor, 5, color, -1)
                        for i in range(x, x + w_box, 15): cv2.line(frame, (i, tracker.tripwire_y), (i + 5, tracker.tripwire_y), (255, 0, 255), 2)
                else: print(f"[{camera_id}] ✗ Tracker ID {tracker.id} ({tracker.name}) removed (disappeared).")
            trackers = active_trackers

            stats = shared_manager.get_stats()
            info_text = f"Entries: {stats['total_entries']} | Exits: {stats['total_exits']} | Occupancy: {stats['occupancy']}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            out.write(frame)
            cv2.imshow(f'Attendance System - {camera_id}', frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        print(f"\n[{camera_id}] ✓ Processing finished.")
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print("\n🎯 DUAL-CAMERA ATTENDANCE SYSTEM (Corrected)")
    
    # Use a manager to create shared objects
    with multiprocessing.Manager() as manager:
        # Create a shared lock for file I/O (both log and embeddings)
        shared_lock = manager.Lock()
        
        # Create shared dictionaries for the attendance manager
        shared_entries = manager.dict()
        shared_exits = manager.dict()
        
        # The manager itself is a proxy and can be passed to processes
        shared_attendance_manager = AttendanceManager(SHARED_LOG_FILE, shared_lock, shared_entries, shared_exits)

        # Create processes, passing the shared manager and lock
        entry_process = multiprocessing.Process(
            target=process_video_stream,
            args=(VIDEO_SOURCE_ENTRY, OUTPUT_VIDEO_ENTRY, "entry", shared_attendance_manager, shared_lock)
        )
        
        exit_process = multiprocessing.Process(
            target=process_video_stream,
            args=(VIDEO_SOURCE_EXIT, OUTPUT_VIDEO_EXIT, "exit", shared_attendance_manager, shared_lock)
        )

        entry_process.start()
        exit_process.start()
        print("\n🔄 Both camera processes started. Press 'q' in any video window to stop.")

        entry_process.join()
        exit_process.join()

        print("\n✓ All processes finished.")
        shared_attendance_manager.print_final_report()