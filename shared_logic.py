# shared_logic.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
import json
from filelock import FileLock # NEW: For safe file writing across scripts

# --- CONFIGURATION (can be overridden by specific scripts) ---
KNOWN_FACES_DIR = 'dataset'
EMBEDDINGS_FILE = 'face_embeddings_final.pkl'
DETECTION_INTERVAL = 10
RESIZE_FACTOR = 0.5
RECOGNITION_THRESHOLD = 0.9
MAX_DISAPPEARED_FRAMES = 20
IOU_THRESHOLD = 0.4
REQUIRED_INPUT_SIZE = (160, 160)
VERTICAL_CROSSING_THRESHOLD_RATIO = 0.15

# --- Models (will be initialized per script) ---
EMBEDDER = None
DETECTOR = None

# --- HELPER FUNCTIONS ---
def calculate_iou(boxA, boxB):
    xA=max(boxA[0],boxB[0]); yA=max(boxA[1],boxB[1]); xB=min(boxA[0]+boxA[2],boxB[0]+boxB[2]); yB=min(boxA[1]+boxA[3],boxB[1]+boxB[3])
    interArea=max(0,xB-xA)*max(0,yB-yA); boxAArea=boxA[2]*boxA[3]; boxBArea=boxB[2]*boxB[3]
    denominator=float(boxAArea+boxBArea-interArea)
    return interArea/denominator if denominator!=0 else 0.0

def extract_face(image, required_size=REQUIRED_INPUT_SIZE):
    image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); results=DETECTOR.detect_faces(image_rgb)
    if not results: return None,None
    x1,y1,width,height=results[0]['box']; x1,y1=abs(x1),abs(y1); x2,y2=x1+width,y1+height
    face=image_rgb[y1:y2,x1:x2]
    return(cv2.resize(face,required_size),(x1,y1,width,height)) if face.size!=0 else(None,None)

def get_embedding(face_pixels):
    face_pixels=face_pixels.astype('float32'); samples=np.expand_dims(face_pixels,axis=0)
    return EMBEDDER.embeddings(samples)[0]

def load_known_faces(directory, embeddings_file):
    # This function is now simpler, as it doesn't need multiprocessing locks
    def get_dataset_modification_time(d):
        latest=0
        if not os.path.exists(d): return latest
        for root,_,files in os.walk(d):
            for file in files: 
                try: latest=max(latest,os.path.getmtime(os.path.join(root,file))); 
                except OSError: continue
        return latest
    
    recompute = not os.path.exists(embeddings_file) or get_dataset_modification_time(directory) > os.path.getmtime(embeddings_file)
    if recompute:
        print("Dataset changed or embeddings not found. Recomputing...")
        known_embeddings={}
        for person_name in os.listdir(directory):
            person_dir=os.path.join(directory,person_name)
            if os.path.isdir(person_dir):
                embeddings_for_person=[]
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.jpg','.jpeg','.png')):
                        path=os.path.join(person_dir,filename); image=cv2.imread(path)
                        if image is not None:
                            face,_=extract_face(image)
                            if face is not None: embeddings_for_person.append(get_embedding(face))
                if embeddings_for_person: known_embeddings[person_name]=np.mean(embeddings_for_person,axis=0)
        with open(embeddings_file,'wb') as f: pickle.dump(known_embeddings,f)
        print(f"✓ Embeddings saved to {embeddings_file}")
        return known_embeddings
    
    print("Using cached embeddings.")
    with open(embeddings_file,'rb') as f: return pickle.load(f)

# In shared_logic.py

def create_tracker():
    """
    Creates a CSRT tracker instance, handling multiple OpenCV versions.
    """
    try:
        # For newer versions (4.5.3+)
        if hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
        
        # For intermediate versions (approx 3.4.2 - 4.5.2)
        # The 'legacy' module was introduced and then removed.
        if hasattr(cv2, 'legacy'):
            return cv2.legacy.TrackerCSRT_create()
            
        # For very old versions (before 3.4.2), it was directly available.
        # This is implicitly covered by the first check, but we keep it clear.
        # If all checks fail, we raise an error.
        
        print("ERROR: Could not find a suitable CSRT tracker in your OpenCV version.")
        print("Please ensure you have 'opencv-contrib-python' installed.")
        return None

    except Exception as e:
        print(f"ERROR: Failed to create tracker. Exception: {e}")
        print("Please ensure you have 'opencv-contrib-python' installed.")
        return None
# --- CLASSES ---
class AttendanceManager:
    def __init__(self, log_file):
        self.log_file = log_file
        self.lock = FileLock(f"{self.log_file}.lock") # Create a lock file

    def record_crossing(self, name, direction, timestamp, camera_id):
        with self.lock: # This ensures only one script can access the file at a time
            # 1. Load the current state from the file
            log_data = {'entries': {}, 'exits': {}}
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    try: log_data = json.load(f)
                    except json.JSONDecodeError: pass # File is empty or corrupt, start fresh
            
            entries = log_data.get('entries', {})
            exits = log_data.get('exits', {})

            # 2. Apply logic
            is_new_record = False
            is_inside = name in entries and name not in exits
            
            if direction == "entry" and not is_inside:
                entries[name] = timestamp.isoformat()
                if name in exits: del exits[name]
                is_new_record = True
                print(f"[{camera_id}] 🚪 ENTRY Recorded: {name} at {timestamp.strftime('%H:%M:%S')}")
            elif direction == "exit" and is_inside:
                exits[name] = timestamp.isoformat()
                is_new_record = True
                print(f"[{camera_id}] 🚪 EXIT Recorded: {name} at {timestamp.strftime('%H:%M:%S')}")

            # 3. Save back to the file if a change was made
            if is_new_record:
                log_data = {'entries': entries, 'exits': exits}
                with open(self.log_file, 'w') as f: json.dump(log_data, f, indent=4)
    
    def get_stats(self):
        with self.lock:
            if not os.path.exists(self.log_file): return {'total_entries':0, 'total_exits':0, 'occupancy':0}
            with open(self.log_file, 'r') as f:
                try: log_data = json.load(f)
                except json.JSONDecodeError: return {'total_entries':0, 'total_exits':0, 'occupancy':0}
            entered=set(log_data.get('entries',{}).keys())
            exited=set(log_data.get('exits',{}).keys())
            return {'total_entries':len(entered),'total_exits':len(exited),'occupancy':len(entered-exited)}

class PersonTracker:
    # (This class is unchanged)
    def __init__(self, tracker_id, name, bbox, frame, vertical_threshold):
        self.id=tracker_id; self.name=name; self.vertical_threshold=vertical_threshold
        self.disappeared_frames=0; self.has_crossed_line=False
        self.tracker=create_tracker(); self.tracker.init(frame,bbox)
        self.last_bbox=bbox
        x,y,w,h=bbox; self.initial_anchor_y=int(y+h); self.current_anchor=(int(x+w/2),int(y+h)); self.tripwire_y=self.initial_anchor_y
    def update(self,frame):
        success,bbox=self.tracker.update(frame)
        if success:
            self.last_bbox=bbox; self.disappeared_frames=0; x,y,w,h=[int(v) for v in bbox]
            self.current_anchor=(int(x+w/2),int(y+h))
            if not self.has_crossed_line: self._check_vertical_crossing()
        else: self.last_bbox=None; self.disappeared_frames+=1
        return self.last_bbox
    def _check_vertical_crossing(self):
        if abs(self.current_anchor[1]-self.initial_anchor_y)>self.vertical_threshold:
            self.has_crossed_line=True; print(f"🚶✅ ID {self.id} ({self.name}) crossed their vertical threshold!")

# --- THE MAIN PROCESSING FUNCTION ---
def process_video_stream(video_source, output_video_path, direction, log_file):
    global DETECTOR, EMBEDDER
    
    camera_id = f"{direction.upper()} CAM"
    print(f"[{camera_id}] Initializing models...")
    DETECTOR = MTCNN()
    EMBEDDER = FaceNet()
    print(f"[{camera_id}] ✓ Models initialized.")
    
    known_faces = load_known_faces(KNOWN_FACES_DIR, EMBEDDINGS_FILE)
    manager = AttendanceManager(log_file)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[{camera_id}] ✗ Error opening video: {video_source}"); return

    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25
    
    vertical_threshold_pixels = int(h * VERTICAL_CROSSING_THRESHOLD_RATIO)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count, next_id, trackers = 0, 0, []
    start_time = datetime.now()

    print(f"[{camera_id}] Starting processing for {video_source}...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # The rest of the loop is the same as before
            current_time = start_time + timedelta(seconds=frame_count / fps)
            
            for tracker in list(trackers):
                tracker.update(frame)
                if tracker.has_crossed_line:
                    manager.record_crossing(tracker.name, direction, current_time, camera_id)
                    trackers.remove(tracker)
                    print(f"[{camera_id}] ✓ Tracker ID {tracker.id} ({tracker.name}) processed and removed.")

            if frame_count % DETECTION_INTERVAL == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
                detections = []
                face_crops_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = DETECTOR.detect_faces(face_crops_rgb)
                for face_info in results:
                    x,y,w_face,h_face=[int(v/RESIZE_FACTOR) for v in face_info['box']]
                    face_crop_bgr=frame[y:y+h_face,x:x+w_face]
                    if face_crop_bgr.size==0: continue
                    face_img,_=extract_face(face_crop_bgr)
                    if face_img is None: continue
                    emb=get_embedding(face_img); min_dist,identity=float('inf'),'Unknown'
                    for name,known_emb in known_faces.items():
                        dist=euclidean(emb,known_emb)
                        if dist<RECOGNITION_THRESHOLD and dist<min_dist: min_dist,identity=dist,name
                    if identity!='Unknown': detections.append({'name':identity,'bbox':(x,y,w_face,h_face)})
            
                matched_tracker_indices=set(); unmatched_detections_indices=list(range(len(detections)))
                if detections:
                    for i,det in enumerate(detections):
                        best_iou,best_match_idx=0,-1
                        for j,tracker in enumerate(trackers):
                            if tracker.last_bbox is None: continue
                            iou=calculate_iou(det['bbox'],tracker.last_bbox)
                            if iou>IOU_THRESHOLD and iou>best_iou and j not in matched_tracker_indices: best_iou,best_match_idx=iou,j
                        if best_match_idx!=-1:
                            trackers[best_match_idx].tracker.init(frame,det['bbox']); trackers[best_match_idx].last_bbox=det['bbox']
                            trackers[best_match_idx].disappeared_frames=0
                            if i in unmatched_detections_indices: unmatched_detections_indices.remove(i)
                            matched_tracker_indices.add(best_match_idx)

                for i in unmatched_detections_indices:
                    det = detections[i]
                    is_already_tracked = any(t.name == det['name'] for t in trackers)
                    if not is_already_tracked:
                        new_tracker = PersonTracker(next_id, det['name'], det['bbox'], frame, vertical_threshold_pixels)
                        trackers.append(new_tracker)
                        print(f"[{camera_id}] ✨ New tracker: ID {next_id} for {det['name']}")
                        next_id += 1

            active_trackers=[]
            for tracker in trackers:
                if tracker.disappeared_frames<MAX_DISAPPEARED_FRAMES:
                    active_trackers.append(tracker)
                    if tracker.last_bbox:
                        x,y,w_box,h_box=[int(v) for v in tracker.last_bbox]; color=(255,255,0)
                        cv2.rectangle(frame,(x,y),(x+w_box,y+h_box),color,2)
                        cv2.putText(frame,f"ID {tracker.id}: {tracker.name}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                        cv2.circle(frame,tracker.current_anchor,5,color,-1)
                        for i in range(x,x+w_box,15): cv2.line(frame,(i,tracker.tripwire_y),(i+5,tracker.tripwire_y),(255,0,255),2)
                else: print(f"[{camera_id}] ✗ Tracker ID {tracker.id} ({tracker.name}) removed (disappeared).")
            trackers=active_trackers

            stats = manager.get_stats()
            info_text=f"Entries:{stats['total_entries']} | Exits:{stats['total_exits']} | Occupancy:{stats['occupancy']}"
            cv2.putText(frame,info_text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),3)
            cv2.putText(frame,info_text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            
            out.write(frame); cv2.imshow(f'Attendance System - {camera_id}',frame)
            frame_count+=1
            if cv2.waitKey(1)&0xFF==ord('q'): break
    finally:
        print(f"\n[{camera_id}] ✓ Processing finished.")
        cap.release(); out.release(); cv2.destroyAllWindows()