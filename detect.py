import cv2



Face_feature = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Eyes_feature = cv2.CascadeClassifier('haarcascade_eye.xml')
Nose_feature = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
Hand_feature = cv2.CascadeClassifier('hand.xml')
Mouth_feature = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

def detect(grey,frame):
    face = Face_feature.detectMultiScale(grey,1.5,5)
    x=1
    if Face_feature.detectMultiScale(grey,1.5,5) is True:
        x=x+1
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)##first image second upper corner third lower fourth rgb
        roi_grey = grey[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = Eyes_feature.detectMultiScale(roi_grey,1.1,3)
        for (ex,ey,ew,eh) in eyes:
            cv2.circle(roi_color,(ex+30,ey+20),20,(255,255,45),2)
        nose = Nose_feature.detectMultiScale(roi_grey,1.1,3)
        for (nx,ny,nw,nh) in nose:    
            cv2.circle(roi_color,(nx+30,ny+10),30,(255,255,255),2)
            mouth = Mouth_feature.detectMultiScale(roi_grey,1.7,3)
        for (mx,my,mw,mh) in mouth:
            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),1.1,2)
        cv2.putText(
        frame, "Face Detected", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
    hand = Hand_feature.detectMultiScale(grey,1.1,3)
    for (hx,hy,hw,hh) in hand:
        cv2.rectangle(frame, (hx,hy),(hx+hw,hy+hh),(255,255,0),2)    

    return frame,x    
