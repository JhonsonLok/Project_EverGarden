import cv2
import time
import numpy as np
from pymouse import PyMouse
from cvzone.HandTrackingModule import HandDetector

# Mouse instantiation
m = PyMouse() 
cx, cy = m.position()

# Initialization 
px, py = 0, 0
smooth = 5 # Smooth Rate
f_red = 200 # Frame Reduction
w_cam, h_cam = 960, 540 # Size of webcam window # Size: (854, 480) (960, 540) (1280, 720)
w_scr, h_scr = m.screen_size() # 获取电脑屏幕大小，这个不用改
click = np.array([0, 1, 1, 0, 0])

# Webcam & Detector
cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
detector = HandDetector(detectionCon=0.9)

while True:
    succes, img = cap.read()
    img = cv2.flip(img,1)
    hands, img = detector.findHands(img)


    if hands:
        # Hand
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detector.fingersUp(hand) #Finds how many fingers are open

        # Cursor  
        cursor = lmList[8]        
        check = np.array(fingers) == click
        count = check.sum()

        if count >= 3:
            cx = cursor[0]
            cy = cursor[1]
            # Transform coordinate
            tx = int(np.interp(cx, (f_red, w_cam - f_red), (0, w_scr)))
            ty = int(np.interp(cy, (f_red, h_cam - f_red), (0, h_scr)))
            fx = px + (tx - px) // smooth
            fy = py + (ty - py) // smooth
        else:
            cx, cy = m.position() 
            fx, fy = cx, cy
        
        #print(lmList[4], lmList[8], lmList[12])

        length1, info1, img1 = detector.findDistance(lmList[4], lmList[8], img)  # with draw
        length2, info2, img2 = detector.findDistance(lmList[4], lmList[12], img)  # with draw
        if length1 < 30:
            m.click(fx, fy)
            time.sleep(0.02)
        if length2 < 15:
            m.click(fx, fy, button=2)
            time.sleep(0.02)   
        if length1 < 30 and length2 < 30:
            m.move(fx, fy)                 


        m.move(fx, fy)
        px, py = fx, fy

    # Display 
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)    
    cv2.imshow("Image",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()