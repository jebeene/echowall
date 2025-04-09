import cv2
import time

from face_storage import load_encodings, load_face_images
from face_detection import detect_and_store_unique_faces
from mural_creator import create_mural
from overlay import add_overlay

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return

    known_encs = load_encodings()
    last_time = 0

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        new_detected = detect_and_store_unique_faces(frame, known_encs)
        if new_detected:
            last_time = time.time()

        faces = load_face_images()
        mural = create_mural(frame, faces)
        mural = add_overlay(mural, len(faces), new_detected, last_time)

        cv2.imshow("Face Mural", mural)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()