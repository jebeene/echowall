import cv2
import time

def add_overlay(frame, face_count, new_face, last_time):
    cv2.putText(frame, f"Faces in Mural: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    if new_face or (time.time() - last_time < 3):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 255, 0), -1)
        cv2.putText(frame, "New Face Detected!", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    return frame