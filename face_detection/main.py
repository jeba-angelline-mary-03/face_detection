import face_recognition
import cv2
import numpy as np

known_image = face_recognition.load_image_file("known_person.jpeg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

known_face_encodings = [
    known_face_encoding,
]
known_face_names = [
    "Known Person",
]

unknown_image = face_recognition.load_image_file("unknown_person2.jpg")

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

rgb_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.rectangle(rgb_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(rgb_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

cv2.imshow('Image', rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()