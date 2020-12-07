import numpy as np
import cv2 as cv
import face_recognition

video_capture = cv.VideoCapture(0, cv.CAP_DSHOW)

# Haar cascade face recognition
haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
people = ['Andreea', 'Black Widow', 'Peter Parker', 'Tony Stark']

# dlib Face Recognition
widow_image = face_recognition.load_image_file('known_people/Black Widow.png')
widow_face_encoding = face_recognition.face_encodings(widow_image)[0]

parker_image = face_recognition.load_image_file('known_people/Peter Parker.jpg')
parker_face_encoding = face_recognition.face_encodings(parker_image)[0]

stark_image = face_recognition.load_image_file('known_people/Tony Stark.jpg')
stark_face_encoding = face_recognition.face_encodings(stark_image)[0]

andreea_image = face_recognition.load_image_file('known_people/Andreea.png')
andreea_face_encoding = face_recognition.face_encodings(andreea_image)[0]

known_face_encodings = [
    widow_face_encoding,
    parker_face_encoding,
    stark_face_encoding,
    andreea_face_encoding
]

known_face_names = [
    "Black Widow",
    "Peter Parker",
    "Tony Stark",
    "Andreea"
]

face_locations = []
face_encodings = []
face_names = []


def haar_cascade_recognition(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(faces_roi)

        # Draw a box around the face
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

        # Draw a label with a name below the face
        cv.rectangle(img, (x, y+h), (x+w, y+h+20), (0, 255, 0), -1)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(img, str(people[label]), (x+2, y+h+15), font, 0.5, (255, 255, 255), 1)

    cv.imshow('Haar Cascade Recognition', img)


def dlib_face_recognition(img):
    rgb_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)

        cv.rectangle(img, (left, bottom), (right, bottom+20), (0, 255, 0), -1)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(img, name, (left+6, bottom+15), font, 0.5, (255, 255, 255), 1)

    cv.imshow('dlib Face Recognition', img)


while True:
    ret, frame = video_capture.read()
    if ret == False:
        break

    ratio = frame.shape[0]/frame.shape[1]
    width = 400
    height = int(width*ratio)
    frame = cv.resize(frame, (width, height))

    haar_cascade_recognition(frame.copy())
    dlib_face_recognition(frame.copy())

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
