import cv2
import dlib
import numpy as np

# Initialize the face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this model (see instructions below)

def distort_face(frame, landmarks):
    """
    Apply advanced distortions to the face by transforming specific regions.
    """
    landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.int32)

    # Create masks for inpainting
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Example 1: Stretch the left cheek
    left_cheek_points = np.array([landmarks_array[i] for i in [0, 1, 2, 3, 31, 50, 49, 48]])
    hull = cv2.convexHull(left_cheek_points)
    cv2.fillConvexPoly(mask, hull, 255)

    # Apply stretch transformation to the left cheek
    left_cheek_center = tuple(np.mean(left_cheek_points, axis=0).astype(float))
    stretch_matrix = cv2.getRotationMatrix2D(left_cheek_center, 0, 1.3)  # Stretch by 1.3x
    stretched_cheek = cv2.warpAffine(frame, stretch_matrix, (frame.shape[1], frame.shape[0]))

    # Inpaint the original left cheek area
    inpainted_frame = cv2.inpaint(frame, mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
    stretched_cheek_mask = cv2.warpAffine(mask, stretch_matrix, (frame.shape[1], frame.shape[0]))
    stretched_cheek_mask_3d = cv2.merge([stretched_cheek_mask] * 3)
    frame = cv2.bitwise_and(inpainted_frame, cv2.bitwise_not(stretched_cheek_mask_3d)) + cv2.bitwise_and(stretched_cheek, stretched_cheek_mask_3d)

    # Example 2: Move the nose
    mask[:] = 0  # Reset the mask
    nose_points = landmarks_array[27:36]  # Indices for the nose
    hull = cv2.convexHull(nose_points)
    cv2.fillConvexPoly(mask, hull, 255)

    # Apply translation to the nose
    translation_matrix = np.float32([[1, 0, 20], [0, 1, 20]])  # Move nose diagonally
    moved_nose = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))

    # Inpaint the original nose area
    inpainted_frame = cv2.inpaint(frame, mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
    moved_nose_mask = cv2.warpAffine(mask, translation_matrix, (frame.shape[1], frame.shape[0]))
    moved_nose_mask_3d = cv2.merge([moved_nose_mask] * 3)
    frame = cv2.bitwise_and(inpainted_frame, cv2.bitwise_not(moved_nose_mask_3d)) + cv2.bitwise_and(moved_nose, moved_nose_mask_3d)

    # Example 3: Move the mouth
    mask[:] = 0  # Reset the mask
    mouth_points = landmarks_array[48:68]  # Indices for the mouth
    hull = cv2.convexHull(mouth_points)
    cv2.fillConvexPoly(mask, hull, 255)

    # Apply translation to the mouth
    translation_matrix = np.float32([[1, 0, -30], [0, 1, 30]])  # Move mouth up and to the left
    moved_mouth = cv2.warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))

    # Inpaint the original mouth area
    inpainted_frame = cv2.inpaint(frame, mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
    moved_mouth_mask = cv2.warpAffine(mask, translation_matrix, (frame.shape[1], frame.shape[0]))
    moved_mouth_mask_3d = cv2.merge([moved_mouth_mask] * 3)
    frame = cv2.bitwise_and(inpainted_frame, cv2.bitwise_not(moved_mouth_mask_3d)) + cv2.bitwise_and(moved_mouth, moved_mouth_mask_3d)

    return frame


def apply_uncanny_effect(frame):
    """
    Detect faces and distort their features.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        frame = distort_face(frame, landmarks)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit the Uncanny Mirror.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame

        # Apply the uncanny effect
        uncanny_frame = apply_uncanny_effect(frame)

        cv2.imshow('Uncanny Mirror', uncanny_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
