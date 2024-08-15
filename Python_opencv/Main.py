import numpy as np
import csv
import face_recognition
import datetime as dt
import cv2

# To capture our face

video_capture = cv2.VideoCapture(0)

# To load face

Devs_face = face_recognition.load_image_file("faces/Devs.jpg")
Devs_encoding = face_recognition.face_encodings(Devs_face)[0]

Smits_face = face_recognition.load_image_file("faces/Smit.jpg")
Smits_encoding = face_recognition.face_encodings(Smits_face)[0]

known_encoding = [Devs_encoding, Smits_encoding]
known_names = ["Devashish", "Smits"]

#expected students

students = known_names.copy()

face_locations = []
face_encodings = []

now = dt.now()
current_date = now.strftime("%Y-%m-%d")

#To create a csv file of date and time

f= open(f"{current_date}.csv","w+",newline="")
lnwriter = csv.writer(f)

#Capture frames

while True :
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encoding, face_encoding)
        face_distance = face_recognition.face_distance(known_encoding,face_encoding)

        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
             name = known_names[best_match_index]

        if name in known_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame+name+" Present",bottomLeftCornerOfText,fontScale,fontColor,thickness,lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name,current_time])



        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video_capture.release()
cv2.destroyAllWindows()
f.close()




