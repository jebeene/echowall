# wacced
**wacced** is an interactive project that creates a dynamic mural from detected faces in a video feed. The program detects faces in real-time, stores _unique_ faces, and generates a mural composed of stored faces. The mural dynamically evolves as new faces are detected.

## Installation

### Prerequisites
1.	Python 3.x: Make sure Python is installed on your system.
2.	Required Libraries: The project depends on several libraries, which you can install using pip or conda.

### Install Dependencies
```
conda install -c conda-forge dlib opencv
pip install face_recognition
```

## Usage

### Run the Script

To run the application:
```python wacced.py```

This will start the webcam feed, and you should see the video feed be generated by the mural of faces, which updates in real-time as new faces are detected.

## Adjusting Real-Time Parameters
You can adjust the following constants to tweak how the mural appears:
1.	Max Stored Faces: This limits how many unique faces are stored in the system. When the number exceeds this value, older faces are deleted.
2.	Tile Color Weight: This controls how much of the tile’s color comes from the video feed versus the face image.
3.	Face Size: Controls the size of each face tile in the mural.
-------
This project is inspired by the [Uncanny Mirror](https://vimeo.com/336559940).
