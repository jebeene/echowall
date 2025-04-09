from pathlib import Path

MAX_STORED_FACES = 500
VIDEO_FEED_COLOR_WEIGHT = 0.7
FACE_SIZE = 5
FACE_DIR = Path("unique_faces")
FACE_DIR.mkdir(exist_ok=True)