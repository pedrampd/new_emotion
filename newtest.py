import cv2 
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
# from dlib import get_frontal_face_detector as face_detector

import imutils

# def draw()


cascade_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
# dt = face_detector()
with open('fer.json') as f:
    loaded_model = f.read()
    
model = model_from_json(loaded_model)

model.load_weights('fer.h5')

face_detection = cv2.CascadeClassifier(cascade_path)

EMOTIONS = ["Angry" ,"Disgust","Scared", "Happy", "Sad", "Surprised",
 "Neutral"]

camera = cv2.VideoCapture(0)

n = 0
new_preds = [0,0,0,0,0,0,0]
new_label = 'Neutral'
# fX,fY,fW,fH = 0,0,0,0
pred_list = []
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

    canvas = np.zeros((500, 800, 3), dtype="uint8")
    canvas += 255
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
                    # Extract the ROI f the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
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
        # pred_list.append(preds)
        # cv2.putText(frameClone,text,(10,40+(22*i)) , cv2.FONT_HERSHEY_SIMPLEX,0.7,(102, 0, 102),2)
        n += 1
        if (n == 6):
            n = 0
            new_preds = preds
            new_label = label        
        cv2.putText(frameClone, new_label, (fX + int(fW/3), fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, (97,136,133), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(255, 255, 255), 3)

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, new_preds)):
		                # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)
		     # w = int(prob * 300)
	      #   cv2.rectangle(canvas, (7, (i * 35) + 5),
	      #       (w, (i * 35) + 35), (0, 0, 255), -1)
	      #   cv2.putText(canvas, text, (10, (i * 35) + 23),
	      #   cv2.FONT_HERSHEY_SIMPLEX, 0.45,
	      #       (255, 255, 255), 2)            
		               

            w = int(prob*750)

            cv2.rectangle(canvas,(7 , (i * 60) +5) , (w,(i*60) + 45) , (212,195,40),-1)
            cv2.putText(canvas,text,(10,(i*60) + 25) , cv2.FONT_HERSHEY_DUPLEX , 0.8 , (0,0,0),2)
		    # old_values.append(prob)                          
    cv2.namedWindow('name',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
# pred_list = np.array(pred_list)
# with open ('preds.txt','w') as f:
# 	f.write(pred_list)