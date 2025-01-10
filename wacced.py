import cv2
import numpy as np
import os
import face_recognition
import random
import time
import uuid

MAX_STORED_FACES = 500  # Maximum number of faces to store
VIDEO_FEED_COLOR_WEIGHT = 0.7 # How much to preserve tile image colors
FACE_SIZE = 5 # How large (pixels) should each face tile be

# Directory to store unique faces
face_storage_dir = "unique_faces"
os.makedirs(face_storage_dir, exist_ok=True)
known_encodings = []

def prune_stored_faces():
    """
    Remove old faces if the number of stored faces exceeds the limit.
    """
    stored_files = [f for f in os.listdir(face_storage_dir) if f.endswith(".png")]
    if len(stored_files) > MAX_STORED_FACES:
        # Sort files by creation time and remove the oldest ones
        stored_files.sort(key=lambda f: os.path.getctime(os.path.join(face_storage_dir, f)))
        for file_to_remove in stored_files[:len(stored_files) - MAX_STORED_FACES]:
            os.remove(os.path.join(face_storage_dir, file_to_remove))

def load_encodings_and_faces():
    """
    Load face images and encodings from the storage directory.
    Ensure encodings and images are synchronized.
    """
    encodings = []
    for filename in os.listdir(face_storage_dir):
        if filename.endswith(".png"):
            face_path = os.path.join(face_storage_dir, filename)
            face_image = face_recognition.load_image_file(face_path)
            encoding = face_recognition.face_encodings(face_image)
            if encoding:
                encodings.append(encoding[0])
    return encodings


def detect_and_store_unique_faces(frame, known_encodings):
    """
    Detect faces in the frame and store only unique ones in the directory.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    new_face_detected = False

    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        # Check if the face is unique
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
        if not any(matches):
            # Add the new face encoding to the known encodings
            known_encodings.append(encoding)

            # Save the face to the directory
            face_image = frame[top:bottom, left:right]
            face_image_resized = cv2.resize(face_image, (50, 50))
            unique_id = str(uuid.uuid4())
            face_path = os.path.join(face_storage_dir, f"face_{unique_id}.png")
            cv2.imwrite(face_path, face_image_resized)

            new_face_detected = True

    # Prune stored faces if they exceed the maximum limit
    prune_stored_faces()

    return new_face_detected

def create_mural(frame, stored_faces):
    """
    Replace the video feed with a mural of stored faces that resembles the original feed.
    """
    if not stored_faces:
        return frame  # If no faces are stored, return the original frame

    rows, cols, _ = frame.shape
    grid_rows = rows // FACE_SIZE
    grid_cols = cols // FACE_SIZE

    # Downscale the frame to a grid for tile mapping
    small_frame = cv2.resize(frame, (grid_cols, grid_rows))
    mural = np.zeros_like(frame)

    for r in range(grid_rows):
        for c in range(grid_cols):
            idx = random.randint(0, len(stored_faces) - 1)

            # Get the color of the tile in the small frame
            tile_color = small_frame[r, c]

            # Resize the stored face to match the tile size
            face = cv2.resize(stored_faces[idx], (FACE_SIZE, FACE_SIZE))

            # Blend the face with the tile color to better match the original
            blended_face = cv2.addWeighted(face, 1 - VIDEO_FEED_COLOR_WEIGHT, np.full_like(face, tile_color), VIDEO_FEED_COLOR_WEIGHT, 0)

            # Place the blended face in the mural
            mural[r*FACE_SIZE:(r+1)*FACE_SIZE, c*FACE_SIZE:(c+1)*FACE_SIZE] = blended_face
            idx += 1

    return mural

def add_labels_and_banner(frame, num_faces, new_face_detected, last_detection_time):
    """
    Add the number of unique faces and a banner for new face detection to the video feed.
    """
    # Add label for the number of faces
    label = f"Faces in Mural: {num_faces}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Add banner for new face detection
    if new_face_detected or (time.time() - last_detection_time < 3):  # Show for 3 seconds
        banner_text = "New Face Detected!"
        banner_color = (0, 255, 0)  # Green banner
        banner_thickness = -1  # Filled rectangle
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), banner_color, banner_thickness)
        cv2.putText(frame, banner_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return frame

def main():
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit")

    # Load previously stored encodings and faces
    global known_encodings
    known_encodings = load_encodings_and_faces()

    # Load stored faces from the directory
    stored_faces = []
    for filename in os.listdir(face_storage_dir):
        if filename.endswith(".png"):
            face_path = os.path.join(face_storage_dir, filename)
            face_image = cv2.imread(face_path)
            stored_faces.append(face_image)

    last_detection_time = 0  # Track time of last new face detection

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame

        # Detect and store unique faces
        new_face_detected = detect_and_store_unique_faces(frame, known_encodings)
        if new_face_detected:
            last_detection_time = time.time()

        # Reload faces in case new ones were added
        stored_faces = []
        for filename in os.listdir(face_storage_dir):
            if filename.endswith(".png"):
                face_path = os.path.join(face_storage_dir, filename)
                face_image = cv2.imread(face_path)
                stored_faces.append(face_image)

        # Create the mural
        mural_frame = create_mural(frame, stored_faces)

        labeled_frame = add_labels_and_banner(mural_frame, len(stored_faces), new_face_detected, last_detection_time)

        cv2.imshow('Face Mural', mural_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
