import dlib
import numpy as np
import cv2


def line(img,p1,p2):
    cv2.line(img,p1,p2,(255,255,255))


def tup(part):
    return (part.x,part.y)


def circle(img,pt1):
    cv2.circle (img, pt1, radius=2,color=(255 * alpha, 255 * alpha, 255 * alpha), thickness=-1)

def draw_circles(img,arr,landmarks):
    for i in arr:
        circle (img, tup (landmarks.part (i)))

def draw_the_thing(img, landmarks):
    array = [0,17,21,22,26,16,45,42,14,12,10,8,6,4,2,36,39,27,33,48,51,54,62,66,57,30,24,19]
    draw_circles(img,array,landmarks)
    line (img, tup (landmarks.part(8)), tup (landmarks.part(4)))
    line (img, tup (landmarks.part (8)), tup (landmarks.part (6)))
    line (img, tup (landmarks.part (8)), tup (landmarks.part (10)))
    line (img, tup (landmarks.part (8)), tup (landmarks.part (12)))
    line (img, tup (landmarks.part (10)), tup (landmarks.part (12)))
    line (img, tup (landmarks.part (4)), tup (landmarks.part (6)))
    line (img, tup (landmarks.part (12)), tup (landmarks.part (54)))
    line (img, tup (landmarks.part (48)), tup (landmarks.part (4)))
    line (img, tup (landmarks.part (48)), tup (landmarks.part (58)))
    line (img, tup (landmarks.part (58)), tup (landmarks.part (56)))
    line (img, tup (landmarks.part (56)), tup (landmarks.part (54)))
    line (img, tup (landmarks.part (54)), tup (landmarks.part (52)))
    line (img, tup (landmarks.part (52)), tup (landmarks.part (50)))
    line (img, tup (landmarks.part (48)), tup (landmarks.part (50)))
    line (img, tup (landmarks.part (51)), tup (landmarks.part (57)))
    line (img, tup (landmarks.part (48)), tup (landmarks.part (50)))
    line (img, tup (landmarks.part (52)), tup (landmarks.part (31)))
    line (img, tup (landmarks.part (54)), tup (landmarks.part (35)))
    line (img, tup (landmarks.part (52)), tup (landmarks.part (50)))
    line (img, tup (landmarks.part (35)), tup (landmarks.part (34)))
    line (img, tup (landmarks.part (31)), tup (landmarks.part (32)))
    line (img, tup (landmarks.part (52)), tup (landmarks.part (50)))
    line (img, tup (landmarks.part (31)), tup (landmarks.part (48)))
    line (img, tup (landmarks.part (34)), tup (landmarks.part (32)))
    line (img, tup (landmarks.part (33)), tup (landmarks.part (32)))
    line (img, tup (landmarks.part (34)), tup (landmarks.part (33)))
    line (img, tup (landmarks.part (35)), tup (landmarks.part (30)))
    line (img, tup (landmarks.part (31)), tup (landmarks.part (30)))
    line (img, tup (landmarks.part (31)), tup (landmarks.part (27)))
    line (img, tup (landmarks.part (35)), tup (landmarks.part (27)))
    line (img, tup (landmarks.part (50)), tup (landmarks.part (35)))
    line (img, tup (landmarks.part (31)), tup (landmarks.part (27)))
    line (img, tup (landmarks.part (22)), tup (landmarks.part (27)))
    line (img, tup (landmarks.part (21)), tup (landmarks.part (27)))
    line (img, tup (landmarks.part (42)), tup (landmarks.part (27)))
    line (img, tup (landmarks.part (39)), tup (landmarks.part (27)))
    line (img, tup (landmarks.part (22)), tup (landmarks.part (24)))
    line (img, tup (landmarks.part (24)), tup (landmarks.part (26)))
    line (img, tup (landmarks.part (21)), tup (landmarks.part (19)))
    line (img, tup (landmarks.part (19)), tup (landmarks.part (17)))
    line (img, tup (landmarks.part (31)), tup (landmarks.part (27)))
    line (img, tup (landmarks.part (42)), tup (landmarks.part (43)))
    line (img, tup (landmarks.part (43)), tup (landmarks.part (44)))
    line (img, tup (landmarks.part (44)), tup (landmarks.part (45)))
    line (img, tup (landmarks.part (45)), tup (landmarks.part (46)))
    line (img, tup (landmarks.part (46)), tup (landmarks.part (47)))
    line (img, tup (landmarks.part (47)), tup (landmarks.part (42)))
    line (img, tup (landmarks.part (45)), tup (landmarks.part (54)))
    line (img, tup (landmarks.part (42)), tup (landmarks.part (43)))
    line (img, tup (landmarks.part (54)), tup (landmarks.part (14)))
    line (img, tup (landmarks.part (54)), tup (landmarks.part (12)))
    line (img, tup (landmarks.part (22)), tup (landmarks.part (21)))
    line (img, tup (landmarks.part (12)), tup (landmarks.part (14)))
    line (img, tup (landmarks.part (14)), tup (landmarks.part (16)))
    line (img, tup (landmarks.part (16)), tup (landmarks.part (26)))
    line (img, tup (landmarks.part (45)), tup (landmarks.part (26)))
    line (img, tup (landmarks.part (45)), tup (landmarks.part (16)))
    line (img, tup (landmarks.part (45)), tup (landmarks.part (14)))
    line (img, tup (landmarks.part (36)), tup (landmarks.part (37)))
    line (img, tup (landmarks.part (37)), tup (landmarks.part (38)))
    line (img, tup (landmarks.part (39)), tup (landmarks.part (40)))
    line (img, tup (landmarks.part (41)), tup (landmarks.part (40)))
    line (img, tup (landmarks.part (41)), tup (landmarks.part (36)))
    line (img, tup (landmarks.part (38)), tup (landmarks.part (39)))
    line (img, tup (landmarks.part (36)), tup (landmarks.part (48)))
    line (img, tup (landmarks.part (36)), tup (landmarks.part (17)))
    line (img, tup (landmarks.part (36)), tup (landmarks.part (0)))
    line (img, tup (landmarks.part (36)), tup (landmarks.part (2)))
    line (img, tup (landmarks.part (17)), tup (landmarks.part (0)))
    line (img, tup (landmarks.part (2)), tup (landmarks.part (0)))
    line (img, tup (landmarks.part (2)), tup (landmarks.part (4)))
    line (img, tup (landmarks.part (48)), tup (landmarks.part (2)))






detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


img = cv2.imread('10.jpeg')
print(type(img))
faces = detector(img,1)
alpha = 1
camera = cv2.VideoCapture(0)
# width = camera.get(cv2.CV_CAP_PROP_FRAME_WIDTH)   # float
# # Get current height of frame
# height = camera.get(cv2.CV_CAP_PROP_FRAME_HEIGHT) # float
#
#
# # Define the codec and create VideoWriter object
# fourcc = cv2.cv.CV_FOURCC(*'X264')
# out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('testvideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
i = 0
cascade_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(cascade_path)
while True:
    i += 1

    frame = camera.read()[1]
    faces=face_detection.detectMultiScale (frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

        (fX, fY, fW, fH) = faces
        # cv2.rectangle (frame, (fX, fY), (fX + fW, fY + fH),
        #                (0, 0, 255), 2)
        if i >= 20:
            # print(i)
            print(len(faces))
            rect = dlib.rectangle(fX, fY, fX+fW, fY+fH)
                # cv2.rectangle(frame,rect,(0, 0, 255), 2)
            landmarks = predictor(frame, rect)
            draw_the_thing(frame,landmarks)
            if i ==30:
                i = 0


    out.write(frame)
    cv2.imshow('asd',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
# for face in faces:
#     landmarks = predictor(img,face)
#     # for i in range(68):
#     #     cv2.circle(img,(landmarks.part(i).x,landmarks.part(i).y),radius=2,color=(255*alpha,255*alpha,255*alpha), thickness=-1)
#     draw_the_thing(img,landmarks)
#
# cv2.imshow('asd',img)
# cv2.waitKey(0)