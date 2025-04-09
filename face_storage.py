import cv2
from config import FACE_DIR, MAX_STORED_FACES
import face_recognition

def prune_stored_faces():
    stored_files = sorted(FACE_DIR.glob("*.png"), key=lambda f: f.stat().st_ctime)
    for file in stored_files[:-MAX_STORED_FACES]:
        file.unlink()

def load_encodings():
    encodings = []
    for face_path in FACE_DIR.glob("*.png"):
        image = face_recognition.load_image_file(str(face_path))
        encoding = face_recognition.face_encodings(image)
        if encoding:
            encodings.append(encoding[0])
    return encodings

def load_face_images():
    return [cv2.imread(str(f)) for f in FACE_DIR.glob("*.png")]