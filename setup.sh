# download face recognition model
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

mkdir models
mv shape_predictor_68_face_landmarks.dat.bz2 models/
bzip2 -d ./models/shape_predictor_68_face_landmarks.dat.bz2

