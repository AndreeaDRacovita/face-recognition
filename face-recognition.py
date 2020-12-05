import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

people = ['Black Widow', 'Peter Parker', 'Tony Stark']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('Faces/validation/tony.jpg')
ratio = img.shape[0]/img.shape[1]
width = 600
height = int(width*ratio)
img = cv.resize(img, (width, height))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    # Draw a box around the face
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Draw a label with a name below the face
    cv.rectangle(img, (x, y+h), (x+w, y+h+35), (0, 255, 0), -1)
    font = cv.FONT_HERSHEY_DUPLEX
    cv.putText(img, str(people[label]), (x+2, y+h+25), font, 0.5, (255, 255, 255), 2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
