import cv2
import numpy as np
import random
from config import FACE_SIZE, VIDEO_FEED_COLOR_WEIGHT

def create_mural(frame, stored_faces):
    if not stored_faces:
        return frame

    rows, cols, _ = frame.shape
    grid_rows, grid_cols = rows // FACE_SIZE, cols // FACE_SIZE
    small_frame = cv2.resize(frame, (grid_cols, grid_rows))
    mural = np.zeros_like(frame)

    for r in range(grid_rows):
        for c in range(grid_cols):
            idx = random.randint(0, len(stored_faces) - 1)
            tile_color = small_frame[r, c]
            face_tile = cv2.resize(stored_faces[idx], (FACE_SIZE, FACE_SIZE))
            blended = cv2.addWeighted(face_tile, 1 - VIDEO_FEED_COLOR_WEIGHT,
                                      np.full_like(face_tile, tile_color), VIDEO_FEED_COLOR_WEIGHT, 0)
            mural[r*FACE_SIZE:(r+1)*FACE_SIZE, c*FACE_SIZE:(c+1)*FACE_SIZE] = blended

    return mural