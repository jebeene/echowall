import cv2
import uuid
import face_recognition
from config import FACE_DIR
from face_storage import prune_stored_faces

def detect_and_store_unique_faces(frame, known_encodings):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)
    new_detected = False

    for encoding, (top, right, bottom, left) in zip(encs, locs):
        if not any(face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)):
            known_encodings.append(encoding)
            face_img = cv2.resize(frame[top:bottom, left:right], (50, 50))
            face_path = FACE_DIR / f"face_{uuid.uuid4()}.png"
            cv2.imwrite(str(face_path), face_img)
            new_detected = True

    prune_stored_faces()
    return new_detected