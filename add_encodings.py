## This Program helps in storing a user's facial encodings in a mysql data base. 
## Press space bar to capture your face encodings for 5 diferent poses - straight, left, right, up, down.


import cv2
import dlib
import scipy.misc
import numpy as np
DEBUG=0
import pymysql

# extsn=[".jpeg",".JPEG",".PNG",".png",".jpg",".JPG"]
face_detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')
shape_predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

def get_face_encodings(path_to_image):
    image = scipy.misc.imread(path_to_image)
    detected_faces = face_detector(image, 1)
    for face in detected_faces:
            print("Confidence: {}".format(face.confidence))
            if(face.confidence > 1):
                shapes_faces = [shape_predictor(image, face.rect)]
                print("Confident")
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

print("Please look into the webcam and press space-bar. Press ESC to exit anytime")
cam = cv2.VideoCapture(0)
img_counter = 0
person_pose = ["straight", "left", "right", "up", "down"]
l =[]
cv2.namedWindow("Image_Capture")
for i in person_pose:
    print("Please look: " + i)
    while True:
        ret, frame = cam.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("Image_Capture", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break   
        if(k%256 == 32):
            img_name = "Image_{}.png".format(i)
            cv2.imwrite("media/"+img_name, frame)
            face_encodings_in_image = get_face_encodings("media/"+img_name)
            if len(face_encodings_in_image) != 1:
                print("Please change image: " + img_name + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
                exit()
            l.append(face_encodings_in_image[0].dumps())
            break
      
cam.release()
cv2.destroyAllWindows()
name = str(input("Enter Name:"))
progress = "Greetings"
db = pymysql.connect(host='localhost', port=3306, user='<WriteName>', passwd='YourPWD', db='tutors')
cur = db.cursor()
cur.execute("INSERT INTO users VALUES (%s,%s,%s,%s,%s,%s,%s);",(name, progress, l[0],l[1],l[2],l[3],l[4]))
cur.close()
db.commit()

cur = db.cursor()
cur.execute("SELECT * FROM users;")
for row in cur:
    print(row[0], row[1], np.loads(row[2]),np.loads(row[3]),np.loads(row[4]),np.loads(row[5]),np.loads(row[6]))
cur.close()


# Creating SQL DataBase and TABLE 
"""
mysql -u root -p
CREATE USER 'agent'@'localhost' IDENTIFIED BY '<YourPWD>';
GRANT ALL PRIVILEGES ON tutors.* TO 'agent'@'localhost';
FLUSH PRIVILEGES;
mysql -u agent -p
CREATE DATABASE tutors;
use tutors;
CREATE TABLE users(name varchar(20), progress varchar(20), straight BLOB, users.left BLOB, users.right BLOB, up BLOB, down BLOB);
DELETE FROM users where name="<SOMENAME FOR TESTING>";
"""