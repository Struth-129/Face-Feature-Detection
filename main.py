import cv2
from detect import detect




video_capture = cv2.VideoCapture(0)

while True:
    _,frame = video_capture.read()   ##Obtaining the second frame of the video 
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   ##Converting color
    canvas,x = detect(grey,frame)
    print(x)
    cv2.namedWindow('Detection',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection',1600,900)
    cv2.imshow('Detection',canvas)  ##display
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.
