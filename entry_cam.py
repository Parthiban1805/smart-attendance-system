# entry_cam.py
import shared_logic
from datetime import datetime

# --- CONFIGURATION FOR ENTRY CAMERA ---
VIDEO_SOURCE = r"C:\Users\sunda\Downloads\Person monitoring POC-20250623T130324Z-1-001\Person monitoring POC\Entry\VID_20250623_142038.mp4"
DIRECTION = "entry"
OUTPUT_VIDEO_PATH = f'output_video_{DIRECTION}_final.mp4'
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