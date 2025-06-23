# exit_cam.py
import shared_logic
from datetime import datetime

# --- CONFIGURATION FOR EXIT CAMERA ---
VIDEO_SOURCE = r"C:\Users\sunda\Downloads\Person monitoring POC-20250623T130324Z-1-001\Person monitoring POC\Exit\VID_20250623_142258.mp4"
DIRECTION = "exit"
OUTPUT_VIDEO_PATH = f'output_video_{DIRECTION}_f.mp4'
# IMPORTANT: This MUST be the same file as in entry_cam.py
SHARED_LOG_FILE = f"attendance_log_{datetime.now().strftime('%Y%m%d')}.json"

if __name__ == '__main__':
    print(f"--- STARTING {DIRECTION.upper()} CAMERA ---")
    
    shared_logic.process_video_stream(
        video_source=VIDEO_SOURCE,
        output_video_path=OUTPUT_VIDEO_PATH,
        direction=DIRECTION,
        log_file=SHARED_LOG_FILE
    )

    print(f"--- {DIRECTION.upper()} CAMERA SCRIPT FINISHED ---")