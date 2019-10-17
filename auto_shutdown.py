import face_recognition
import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)


your_image = face_recognition.load_image_file("xxx.jpg")
your_face_encoding = face_recognition.face_encodings(your_image)[0]


known_face_encodings = [
    your_face_encoding
]
known_face_names = [
    "___your_name___"
]

while True:
  
    ret, frame = video_capture.read()

   
    rgb_frame = frame[:, :, ::-1]

    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if matches[0] != True:
            print("Winsows will shut down in 5 seconds")
            os.system("shutdown /s /t 5")
        name = "Unknown"

        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    
    cv2.imshow('Video', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
