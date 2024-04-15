import cv2
import numpy as np
import os 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Mr. Sophal ', 'Bro Seth ', 'Kiyosaki Picture', 'Messi Picture ', 'Elon Musk Picture', 'Bro Toury', 'Bro Vatanak'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)

cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
#w11=cam.set(3)
#h11=cam.set(4)

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
#minW = 1*cam.get(3)
#minH = 1*cam.get(4)
out = cv2.VideoWriter('outpy1test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (int(minW),int(minH)))
while True:
    ret, frame =cam.read()
  
    img = cv2.flip(frame, 1) # Flip vertically
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    out.write(frame)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    frame, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (85,100,250), 
                    2
                   )
        cv2.putText(
                    frame, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    
    cv2.imshow('camera',frame) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
#======================
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output1.mp4',fourcc, 15.0, (int(minW),int(minH)))
#out1 = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))
#======================


# When everything done, release the video capture and video write objects

out.release()

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
#video_writer.release()
cv2.destroyAllWindows()
