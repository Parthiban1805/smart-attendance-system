# 🎯 Smart Attendance System with Dual Camera Setup

An advanced face recognition-based attendance system that monitors both entry and exit points simultaneously using dual cameras. The system tracks people crossing virtual tripwires and maintains real-time attendance logs.

## 🚀 Features

- **Dual Camera Monitoring**: Simultaneous entry and exit tracking
- **Face Recognition**: Uses FaceNet and MTCNN for accurate identification
- **Real-time Tracking**: Object tracking with crossing detection
- **Attendance Logging**: Automatic JSON log generation with timestamps
- **Live Video Output**: Processed video files with visual annotations
- **Occupancy Counting**: Real-time occupancy status

## 📋 Prerequisites

- Python 3.7 or higher
- Two cameras or video files (one for entry, one for exit)
- Dataset of known faces organized in folders

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Parthiban1805/smart-attendance-system.git
cd smart-attendance-system
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Your Dataset
Create a `dataset1` folder with subfolders for each person:
```
dataset1/
├── john_doe/
│   ├── john1.jpg
│   ├── john2.jpg
│   └── john3.jpg
├── jane_smith/
│   ├── jane1.jpg
│   └── jane2.jpg
└── ...
```

### 4. Generate Face Embeddings
```bash
python train.py
```

## 🚦 Usage

### Running the Dual Camera System

**Important**: You need to run both entry and exit cameras simultaneously for complete attendance tracking.

#### Option 1: Using Command Prompt/Terminal (Recommended)

**Windows:**
```cmd
# Open first command prompt for entry camera
python entry_cam.py

# Open second command prompt for exit camera  
python exit_cam.py
```

**Linux/Mac:**
```bash
# Terminal 1 - Entry Camera
python3 entry_cam.py

# Terminal 2 - Exit Camera
python3 exit_cam.py
```


### Configuration

Edit the configuration section in both `entry_cam.py` and `exit_cam.py`:

```python
# --- CONFIGURATION ---
VIDEO_SOURCE = 0  # Use 0 for webcam, or path to video file
KNOWN_FACES_DIR = 'dataset1'
OUTPUT_VIDEO_PATH = 'output_entry.mp4'  # Different for each camera
RECOGNITION_THRESHOLD = 0.9
VERTICAL_CROSSING_THRESHOLD_RATIO = 0.15
ENTRY_IS_DOWN = True  # Set direction for each camera


## 📊 Output Files

The system generates several output files:

- **Video Files**: `output_entry.mp4`, `output_exit.mp4` - Processed videos with annotations
- **Attendance Logs**: `attendance_log_YYYYMMDD_HHMMSS.json` - JSON files with entry/exit records
- **Embeddings**: `face_embeddings.pkl` - Cached face embeddings for faster processing

### Sample Attendance Log
json
{
    "entries": {
        "john_doe": "2025-06-23T09:15:30",
        "jane_smith": "2025-06-23T09:17:45"
    },
    "exits": {
        "john_doe": "2025-06-23T17:30:15"
    }
}
```

## 🎛️ System Controls

- **ESC/Q**: Quit the application
- **Space**: Pause/Resume processing
- **R**: Reset tracking data

**⚠️ Important**: Always run both `entry_cam.py` and `exit_cam.py` simultaneously for accurate attendance tracking. The system relies on both entry and exit points to maintain correct occupancy counts.
