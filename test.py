import cv2 
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from dlib import get_frontal_face_detector as face_detector

import imutils

cascade_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
dt = face_detector()
with open('fer.json') as f:
    loaded_model = f.read()
    
model = model_from_json(loaded_model)

model.load_weights('fer.h5')

face_detection = cv2.CascadeClassifier(cascade_path)

EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

camera = cv2.VideoCapture(0)

n = 0

# fX,fY,fW,fH = 0,0,0,0
while True:
    frame = camera.read()[1]
    #reading the frame
    n += 1
    frame = imutils.resize(frame,width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    # faces = dt(gray,0)
    # print(fX,fY,fH,fW)
    frameClone = frame.copy()
    
 
	
    							
    
    if len(faces) > 0:
    	# fX = faces[0].left()
    	# fY = faces[0].top()
    	# fW = faces[0].right()
    	# fH = faces[0].bottom()
	    
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi -= np.mean(roi)
        roi /= np.std(roi)
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        preds = model.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]


	for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
	                # construct the label text
	    text = "{}: {:.2f}%".format(emotion, prob * 100)
	     # w = int(prob * 300)
      #   cv2.rectangle(canvas, (7, (i * 35) + 5),
      #       (w, (i * 35) + 35), (0, 0, 255), -1)
      #   cv2.putText(canvas, text, (10, (i * 35) + 23),
      #   cv2.FONT_HERSHEY_SIMPLEX, 0.45,
      #       (255, 255, 255), 2)            
	                
	    cv2.putText(frameClone,text,(10,40+(22*i)) , cv2.FONT_HERSHEY_SIMPLEX,0.7,(102, 0, 102),2)
	    cv2.putText(frameClone, label, (fX, fY - 10),
	    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 68, 204), 2)
	    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(26, 255, 26), 3)

	    w = int(prob*300)

	    cv2.rectangle(canvas,(7 , (i * 35) +5) , (w,(i*35) + 35) , (0,0,255),-1)
	    cv2.putText(canvas,text,(10,(i*35) + 23) , cv2.FONT_HERSHEY_SIMPLEX , 0.45 , (255,255,255),2)
	                              

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()